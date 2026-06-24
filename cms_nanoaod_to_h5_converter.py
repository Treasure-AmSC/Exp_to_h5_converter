#!/usr/bin/env python3
"""
CMS NanoAOD → HDF5 Converter  (v2.1 — cross-experiment edition)
================================================================

Converts CMS Run 2 Open Data (NanoAODv9 or PFNanoAODv1 format) into analysis-ready HDF5
files matching the same schema as the ATLAS PHYSLITE converter. Both
experiments produce identical /common/ keys so the foundation model can
train on combined ATLAS+CMS data with experiment_id as a conditioning token.

HDF5 Layout
-----------
  /common/    — Cross-experiment variables (same keys as ATLAS converter).
                All energies/momenta in GeV, angles in radians.
  /cms/       — CMS-specific variables (MVA ID scores, DeepJet, etc.).
  /truth/     — MC truth (GenPart, GenJet, GenMET). Only present for simulation.
  /metadata   — Converter version, dataset info, selection cuts.

Schema
------
  common/electrons : pt, eta, phi, charge, trk_iso03, mask, n
  common/muons     : pt, eta, phi, charge, trk_iso03, mask, n
  common/taus      : pt, eta, phi, charge, is_1prong, mask, n
  common/photons   : pt, eta, phi, trk_iso03, mask, n
  common/jets      : pt, eta, phi, mass, n_trk, mask, n
  common/tracks    : pt, eta, phi, d0, z0, mask, n  (from PFCands if PFNano)
  common/met       : pt, phi, sumet
  common/event     : pvx, pvy, pvz, mu, experiment_id, is_simulation

  cms/electrons    : mvaIso, mvaIso_WP80, mvaIso_WP90, mvaIso_WPL,
                     cutBased, pfRelIso03_all, dxy, dz, mass
  cms/muons        : looseId, mediumId, tightId, pfRelIso04_all,
                     pfRelIso03_all, dxy, dz
  cms/taus         : idDeepTau2017v2p1VSjet, idDeepTau2017v2p1VSe,
                     idDeepTau2017v2p1VSmu, decayMode, rawDeepTau2017v2p1VSjet
  cms/photons      : cutBased, mvaID, mvaID_WP80, mvaID_WP90,
                     pfRelIso03_all
  cms/jets         : btagDeepFlavB, btagDeepFlavC, btagDeepFlavQG,
                     jetId, puId, qgl, nConstituents, nMuons
  cms/pfcands      : pt, eta, phi, mass, charge, pdgId, d0, dz,
                     puppiWeight, trkChi2, trkQuality, n, mask
  cms/event        : event, run, luminosityBlock

  truth/electrons  : pt, eta, phi, mass, pdgId, status, n, mask
  truth/muons      : pt, eta, phi, mass, pdgId, status, n, mask
  truth/taus       : pt, eta, phi, mass, pdgId, status, n, mask
  truth/jets       : pt, eta, phi, mass, n, mask   (GenJet AK4)
  truth/met        : pt, phi

Object Selection
----------------
  Electrons:  pT > 7 GeV, |eta| < 2.5, mvaFall17V2Iso_WPL (loose MVA ID)
  Muons:      pT > 5 GeV, |eta| < 2.4, looseId
  Photons:    pT > 10 GeV, |eta| < 2.5, cutBased >= 1 (loose)
  Taus:       pT > 20 GeV, |eta| < 2.3, idDeepTau2017v2p1VSjet >= 2 (VVLoose)
  Jets:       pT > 30 GeV, |eta| < 4.7, jetId >= 2 (tight)

Isolation mapping (trk_iso03)
-----------------------------
  Electrons: pfRelIso03_all  (PF-based, ΔR < 0.3)
  Muons:     pfRelIso03_all  (PF-based, ΔR < 0.3)
  Photons:   pfRelIso03_all
  (Not identical to ATLAS track-only isolation, but conceptually similar.
   Use experiment_id conditioning to account for the difference.)

PFCands (PFNano only)
---------------------
  When PFCands branches are present (PFNano format), the converter:
    - Fills common/tracks with charged PFCands (pT > 0.5 GeV, pT-sorted),
      matching the ATLAS track schema (pt, eta, phi, d0, z0).
    - Stores all PFCands (charged + neutral, pT > 0.2 GeV) in cms/pfcands.
  When PFCands are absent (standard NanoAOD), common/tracks is left empty.

Input format
------------
  CMS NanoAODv9 or PFNanoAODv1 ROOT files from CERN Open Data Portal.
  Tree name: "Events"
  All branches use flat arrays: nElectron, Electron_pt[nElectron], etc.

Usage
-----
  python cms_nanoaod_to_h5_converter.py input.root -o output.h5
  python cms_nanoaod_to_h5_converter.py data/*.root -o combined.h5 --max-events 100000
"""

import numpy as np
import h5py
import uproot
import awkward as ak
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import json
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


CONVERTER_VERSION = "2.2.0"
EXPERIMENT_ID  = 1          # 0 = ATLAS, 1 = CMS

CONVERTER_CHANGELOG = {
    "2.2.0": "Add PFNano support: PFCands -> common/tracks (charged) + cms/pfcands (all). "
             "Fallback to empty tracks for standard NanoAOD.",
    "2.1.0": "Initial CMS NanoAODv9 converter. Cross-experiment format matching "
             "ATLAS converter v2.1.0. Loose quality cuts, truth particles, metadata.",
}

GeV = 1.0  # NanoAOD already stores pT in GeV


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class CMSObjectConfig:
    electron_pt_cut: float = 7.0      # GeV
    electron_eta_cut: float = 2.5
    electron_require_mva_loose: bool = True  # mvaFall17V2Iso_WPL

    muon_pt_cut: float = 5.0          # GeV
    muon_eta_cut: float = 2.4
    muon_require_loose: bool = True    # looseId

    photon_pt_cut: float = 10.0       # GeV
    photon_eta_cut: float = 2.5
    photon_min_cutbased: int = 1      # >= 1 = loose

    tau_pt_cut: float = 20.0          # GeV
    tau_eta_cut: float = 2.3
    tau_min_deeptau_vsjet: int = 2    # >= 2 = VVLoose

    jet_pt_cut: float = 30.0          # GeV
    jet_eta_cut: float = 4.7
    jet_min_id: int = 2               # >= 2 = tight

    pfcand_pt_cut: float = 0.5        # GeV (charged PFCands for common/tracks)
    pfcand_pt_cut_all: float = 0.2    # GeV (all PFCands for cms/pfcands)

    max_objects: int = 20
    max_tracks: int = 50               # charged PFCands → common/tracks
    max_pfcands: int = 500             # all PFCands → cms/pfcands


# ── Branch lists ──────────────────────────────────────────────────────────────

ELECTRON_BRANCHES = [
    "nElectron",
    "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass",
    "Electron_charge",
    "Electron_pfRelIso03_all",
    "Electron_mvaFall17V2Iso", "Electron_mvaFall17V2Iso_WP80",
    "Electron_mvaFall17V2Iso_WP90", "Electron_mvaFall17V2Iso_WPL",
    "Electron_cutBased",
    "Electron_dxy", "Electron_dz",
]

MUON_BRANCHES = [
    "nMuon",
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass",
    "Muon_charge",
    "Muon_looseId", "Muon_mediumId", "Muon_tightId",
    "Muon_pfRelIso03_all", "Muon_pfRelIso04_all",
    "Muon_dxy", "Muon_dz",
]

PHOTON_BRANCHES = [
    "nPhoton",
    "Photon_pt", "Photon_eta", "Photon_phi", "Photon_mass",
    "Photon_pfRelIso03_all",
    "Photon_cutBased", "Photon_mvaID", "Photon_mvaID_WP80", "Photon_mvaID_WP90",
]

TAU_BRANCHES = [
    "nTau",
    "Tau_pt", "Tau_eta", "Tau_phi", "Tau_mass",
    "Tau_charge", "Tau_decayMode",
    "Tau_idDeepTau2017v2p1VSjet", "Tau_idDeepTau2017v2p1VSe",
    "Tau_idDeepTau2017v2p1VSmu",
    "Tau_rawDeepTau2017v2p1VSjet",
]

JET_BRANCHES = [
    "nJet",
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_jetId", "Jet_puId",
    "Jet_btagDeepFlavB", "Jet_btagDeepFlavC", "Jet_btagDeepFlavQG",
    "Jet_nConstituents", "Jet_nMuons",
    "Jet_qgl",
]

MET_BRANCHES = [
    "MET_pt", "MET_phi", "MET_sumEt",
]

PFCAND_BRANCHES = [
    "nPFCands",
    "PFCands_pt", "PFCands_eta", "PFCands_phi", "PFCands_mass",
    "PFCands_charge", "PFCands_pdgId",
    "PFCands_d0", "PFCands_dz",
    "PFCands_puppiWeight",
    "PFCands_trkChi2", "PFCands_trkQuality",
]

EVENT_BRANCHES = [
    "event", "run", "luminosityBlock",
    "PV_x", "PV_y", "PV_z",
    "PV_npvsGood",
    "Pileup_nTrueInt",  # MC only
]

# Truth (MC only)
GEN_BRANCHES = [
    "nGenPart",
    "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
    "GenPart_pdgId", "GenPart_status", "GenPart_statusFlags",
    "GenPart_genPartIdxMother",
    "nGenJet",
    "GenJet_pt", "GenJet_eta", "GenJet_phi", "GenJet_mass",
    "GenMET_pt", "GenMET_phi",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(ev, name):
    """Get branch from event, return None if not found."""
    return ev[name] if name in ev.fields else None


def _get_available(tree, branch_list):
    keys = set(tree.keys())
    avail = [b for b in branch_list if b in keys]
    missing = [b for b in branch_list if b not in keys]
    if missing:
        logger.debug(f"Branches not found: {missing}")
    return avail


# ── Converter ─────────────────────────────────────────────────────────────────

class CMSNanoAODConverter:

    def __init__(self, config: Optional[CMSObjectConfig] = None):
        self.config = config or CMSObjectConfig()

    def process_file(self, input_file: str) -> Dict:
        """Read a NanoAOD ROOT file and return structured data dict."""
        logger.info(f"Reading: {input_file}")

        tree = uproot.open(f"{input_file}:Events")

        all_branches = []
        for bl in [ELECTRON_BRANCHES, MUON_BRANCHES, PHOTON_BRANCHES,
                   TAU_BRANCHES, JET_BRANCHES, PFCAND_BRANCHES,
                   MET_BRANCHES, EVENT_BRANCHES, GEN_BRANCHES]:
            all_branches.extend(_get_available(tree, bl))

        events = tree.arrays(all_branches, library="ak")
        n = len(events)
        logger.info(f"  {n} events loaded")

        has_truth = "nGenPart" in events.fields

        has_pfcands = "nPFCands" in events.fields

        result = {
            'electrons': self._process_electrons(events),
            'muons':     self._process_muons(events),
            'photons':   self._process_photons(events),
            'taus':      self._process_taus(events),
            'jets':      self._process_jets(events),
            'pfcands':   self._process_pfcands(events) if has_pfcands else None,
            'tracks':    self._process_tracks(events),
            'met':       self._process_met(events),
            'event_info': self._process_event_info(events),
        }

        if has_truth:
            result['truth'] = self._process_truth(events)

        return result

    # ── Object processors ─────────────────────────────────────────────────

    def _process_electrons(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'charge', 'trk_iso03']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['mvaIso', 'mvaIso_WP80', 'mvaIso_WP90', 'mvaIso_WPL',
                         'cutBased', 'pfRelIso03_all', 'dxy', 'dz', 'mass']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Electron_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Electron_eta"])

            # Selection
            pm = (pt_np > self.config.electron_pt_cut) & (np.abs(eta_np) < self.config.electron_eta_cut)
            if self.config.electron_require_mva_loose:
                wpl = _safe(ev, "Electron_mvaFall17V2Iso_WPL")
                if wpl is not None:
                    pm = pm & (ak.to_numpy(wpl) > 0)

            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            c['pt'][i, :n]     = pt_np[si]
            c['eta'][i, :n]    = eta_np[si]
            c['phi'][i, :n]    = ak.to_numpy(ev["Electron_phi"])[si]
            ch = _safe(ev, "Electron_charge")
            if ch is not None: c['charge'][i, :n] = ak.to_numpy(ch)[si]
            iso = _safe(ev, "Electron_pfRelIso03_all")
            if iso is not None: c['trk_iso03'][i, :n] = ak.to_numpy(iso)[si]
            c['n'][i] = n
            c['mask'][i, :n] = True

            # CMS-specific
            for key, branch in [('mvaIso', 'Electron_mvaFall17V2Iso'),
                                ('mvaIso_WP80', 'Electron_mvaFall17V2Iso_WP80'),
                                ('mvaIso_WP90', 'Electron_mvaFall17V2Iso_WP90'),
                                ('mvaIso_WPL', 'Electron_mvaFall17V2Iso_WPL'),
                                ('cutBased', 'Electron_cutBased'),
                                ('pfRelIso03_all', 'Electron_pfRelIso03_all'),
                                ('dxy', 'Electron_dxy'), ('dz', 'Electron_dz'),
                                ('mass', 'Electron_mass')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_muons(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'charge', 'trk_iso03']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['looseId', 'mediumId', 'tightId',
                         'pfRelIso03_all', 'pfRelIso04_all', 'dxy', 'dz']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Muon_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Muon_eta"])

            pm = (pt_np > self.config.muon_pt_cut) & (np.abs(eta_np) < self.config.muon_eta_cut)
            if self.config.muon_require_loose:
                lid = _safe(ev, "Muon_looseId")
                if lid is not None:
                    pm = pm & (ak.to_numpy(lid) > 0)

            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            c['pt'][i, :n]     = pt_np[si]
            c['eta'][i, :n]    = eta_np[si]
            c['phi'][i, :n]    = ak.to_numpy(ev["Muon_phi"])[si]
            ch = _safe(ev, "Muon_charge")
            if ch is not None: c['charge'][i, :n] = ak.to_numpy(ch)[si]
            iso = _safe(ev, "Muon_pfRelIso03_all")
            if iso is not None: c['trk_iso03'][i, :n] = ak.to_numpy(iso)[si]
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('looseId', 'Muon_looseId'),
                                ('mediumId', 'Muon_mediumId'),
                                ('tightId', 'Muon_tightId'),
                                ('pfRelIso03_all', 'Muon_pfRelIso03_all'),
                                ('pfRelIso04_all', 'Muon_pfRelIso04_all'),
                                ('dxy', 'Muon_dxy'), ('dz', 'Muon_dz')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_photons(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'trk_iso03']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['cutBased', 'mvaID', 'mvaID_WP80', 'mvaID_WP90',
                         'pfRelIso03_all']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Photon_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Photon_eta"])

            pm = (pt_np > self.config.photon_pt_cut) & (np.abs(eta_np) < self.config.photon_eta_cut)
            if self.config.photon_min_cutbased > 0:
                cb = _safe(ev, "Photon_cutBased")
                if cb is not None:
                    pm = pm & (ak.to_numpy(cb) >= self.config.photon_min_cutbased)

            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            c['pt'][i, :n]  = pt_np[si]
            c['eta'][i, :n] = eta_np[si]
            c['phi'][i, :n] = ak.to_numpy(ev["Photon_phi"])[si]
            iso = _safe(ev, "Photon_pfRelIso03_all")
            if iso is not None: c['trk_iso03'][i, :n] = ak.to_numpy(iso)[si]
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('cutBased', 'Photon_cutBased'),
                                ('mvaID', 'Photon_mvaID'),
                                ('mvaID_WP80', 'Photon_mvaID_WP80'),
                                ('mvaID_WP90', 'Photon_mvaID_WP90'),
                                ('pfRelIso03_all', 'Photon_pfRelIso03_all')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_taus(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'charge', 'is_1prong']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['idDeepTau2017v2p1VSjet', 'idDeepTau2017v2p1VSe',
                         'idDeepTau2017v2p1VSmu', 'decayMode',
                         'rawDeepTau2017v2p1VSjet']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Tau_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Tau_eta"])

            pm = (pt_np > self.config.tau_pt_cut) & (np.abs(eta_np) < self.config.tau_eta_cut)
            if self.config.tau_min_deeptau_vsjet > 0:
                dt = _safe(ev, "Tau_idDeepTau2017v2p1VSjet")
                if dt is not None:
                    pm = pm & (ak.to_numpy(dt) >= self.config.tau_min_deeptau_vsjet)

            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            c['pt'][i, :n]   = pt_np[si]
            c['eta'][i, :n]  = eta_np[si]
            c['phi'][i, :n]  = ak.to_numpy(ev["Tau_phi"])[si]
            ch = _safe(ev, "Tau_charge")
            if ch is not None: c['charge'][i, :n] = ak.to_numpy(ch)[si]
            dm = _safe(ev, "Tau_decayMode")
            if dm is not None:
                dm_np = ak.to_numpy(dm)[si]
                c['is_1prong'][i, :n] = (dm_np <= 2).astype(np.float32)
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('idDeepTau2017v2p1VSjet', 'Tau_idDeepTau2017v2p1VSjet'),
                                ('idDeepTau2017v2p1VSe', 'Tau_idDeepTau2017v2p1VSe'),
                                ('idDeepTau2017v2p1VSmu', 'Tau_idDeepTau2017v2p1VSmu'),
                                ('decayMode', 'Tau_decayMode'),
                                ('rawDeepTau2017v2p1VSjet', 'Tau_rawDeepTau2017v2p1VSjet')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_jets(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'mass', 'n_trk']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['btagDeepFlavB', 'btagDeepFlavC', 'btagDeepFlavQG',
                         'jetId', 'puId', 'qgl', 'nConstituents', 'nMuons']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Jet_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Jet_eta"])

            pm = (pt_np > self.config.jet_pt_cut) & (np.abs(eta_np) < self.config.jet_eta_cut)
            if self.config.jet_min_id > 0:
                jid = _safe(ev, "Jet_jetId")
                if jid is not None:
                    pm = pm & (ak.to_numpy(jid) >= self.config.jet_min_id)

            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            c['pt'][i, :n]   = pt_np[si]
            c['eta'][i, :n]  = eta_np[si]
            c['phi'][i, :n]  = ak.to_numpy(ev["Jet_phi"])[si]
            m = _safe(ev, "Jet_mass")
            if m is not None: c['mass'][i, :n] = ak.to_numpy(m)[si]
            nc = _safe(ev, "Jet_nConstituents")
            if nc is not None: c['n_trk'][i, :n] = ak.to_numpy(nc)[si]
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('btagDeepFlavB', 'Jet_btagDeepFlavB'),
                                ('btagDeepFlavC', 'Jet_btagDeepFlavC'),
                                ('btagDeepFlavQG', 'Jet_btagDeepFlavQG'),
                                ('jetId', 'Jet_jetId'), ('puId', 'Jet_puId'),
                                ('qgl', 'Jet_qgl'),
                                ('nConstituents', 'Jet_nConstituents'),
                                ('nMuons', 'Jet_nMuons')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_pfcands(self, events) -> Dict:
        """Process PFCands from PFNano into cms/pfcands (all PFCands, charged + neutral)."""
        logger.info("Processing PFCands...")
        n_events = len(events)
        mx = self.config.max_pfcands

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['pt', 'eta', 'phi', 'mass', 'charge', 'pdgId',
                          'd0', 'dz', 'puppiWeight', 'trkChi2', 'trkQuality']}
        cms['n'] = np.zeros(n_events, dtype=np.int32)
        cms['mask'] = np.zeros((n_events, mx), dtype=bool)

        for i, ev in enumerate(events):
            pt = _safe(ev, "PFCands_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)

            pm = pt_np > self.config.pfcand_pt_cut_all
            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            cms['pt'][i, :n]  = pt_np[si]
            cms['eta'][i, :n] = ak.to_numpy(ev["PFCands_eta"])[si]
            cms['phi'][i, :n] = ak.to_numpy(ev["PFCands_phi"])[si]
            cms['n'][i] = n
            cms['mask'][i, :n] = True

            for key, branch in [('mass', 'PFCands_mass'),
                                ('charge', 'PFCands_charge'),
                                ('pdgId', 'PFCands_pdgId'),
                                ('d0', 'PFCands_d0'), ('dz', 'PFCands_dz'),
                                ('puppiWeight', 'PFCands_puppiWeight'),
                                ('trkChi2', 'PFCands_trkChi2'),
                                ('trkQuality', 'PFCands_trkQuality')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        logger.info(f"  PFCands: avg {cms['n'].mean():.0f} per event (max_pfcands={mx})")
        return {'cms': cms}

    def _process_tracks(self, events) -> Dict:
        """Fill common/tracks from charged PFCands (PFNano) or leave empty (NanoAOD)."""
        n_events = len(events)
        mx = self.config.max_tracks
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'd0', 'z0']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        if "nPFCands" not in events.fields:
            return {'common': c, 'cms': {}}

        logger.info("Filling common/tracks from charged PFCands...")
        for i, ev in enumerate(events):
            pt = _safe(ev, "PFCands_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            ch = _safe(ev, "PFCands_charge")
            if ch is None:
                continue
            ch_np = ak.to_numpy(ch)

            pm = (pt_np > self.config.pfcand_pt_cut) & (ch_np != 0)
            idx = np.where(pm)[0]
            n = min(len(idx), mx)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]

            c['pt'][i, :n]  = pt_np[si]
            c['eta'][i, :n] = ak.to_numpy(ev["PFCands_eta"])[si]
            c['phi'][i, :n] = ak.to_numpy(ev["PFCands_phi"])[si]
            d0 = _safe(ev, "PFCands_d0")
            if d0 is not None: c['d0'][i, :n] = ak.to_numpy(d0)[si]
            dz = _safe(ev, "PFCands_dz")
            if dz is not None: c['z0'][i, :n] = ak.to_numpy(dz)[si]
            c['n'][i] = n
            c['mask'][i, :n] = True

        logger.info(f"  Tracks (charged PFCands): avg {c['n'].mean():.0f} per event")
        return {'common': c, 'cms': {}}

    def _process_met(self, events) -> Dict:
        n_events = len(events)
        c = {
            'pt':    np.zeros(n_events, dtype=np.float32),
            'phi':   np.zeros(n_events, dtype=np.float32),
            'sumet': np.zeros(n_events, dtype=np.float32),
        }
        met_pt = _safe(events, "MET_pt")
        if met_pt is not None:
            c['pt'] = ak.to_numpy(met_pt).astype(np.float32)
        met_phi = _safe(events, "MET_phi")
        if met_phi is not None:
            c['phi'] = ak.to_numpy(met_phi).astype(np.float32)
        met_sumet = _safe(events, "MET_sumEt")
        if met_sumet is not None:
            c['sumet'] = ak.to_numpy(met_sumet).astype(np.float32)
        return c

    def _process_event_info(self, events) -> Dict:
        n_events = len(events)
        c = {
            'pvx':           np.zeros(n_events, dtype=np.float32),
            'pvy':           np.zeros(n_events, dtype=np.float32),
            'pvz':           np.zeros(n_events, dtype=np.float32),
            'mu':            np.zeros(n_events, dtype=np.float32),
            'experiment_id': np.ones(n_events, dtype=np.int32) * EXPERIMENT_ID,
            'is_simulation': np.zeros(n_events, dtype=np.int32),
        }
        cms_ev = {
            'event':            np.zeros(n_events, dtype=np.int64),
            'run':              np.zeros(n_events, dtype=np.int32),
            'luminosityBlock':  np.zeros(n_events, dtype=np.int32),
        }

        # PV
        for key, branch in [('pvx', 'PV_x'), ('pvy', 'PV_y'), ('pvz', 'PV_z')]:
            v = _safe(events, branch)
            if v is not None:
                c[key] = ak.to_numpy(v).astype(np.float32)

        # Pileup (MC) or nPVGood (data proxy)
        pu = _safe(events, "Pileup_nTrueInt")
        if pu is not None:
            c['mu'] = ak.to_numpy(pu).astype(np.float32)
            c['is_simulation'] = np.ones(n_events, dtype=np.int32)
        else:
            npv = _safe(events, "PV_npvsGood")
            if npv is not None:
                c['mu'] = ak.to_numpy(npv).astype(np.float32)

        # Event IDs
        for key, branch in [('event', 'event'), ('run', 'run'),
                            ('luminosityBlock', 'luminosityBlock')]:
            v = _safe(events, branch)
            if v is not None:
                cms_ev[key] = ak.to_numpy(v)

        return {'common': c, 'cms': cms_ev}

    def _process_truth(self, events) -> Dict:
        """Process GenPart, GenJet, GenMET."""
        logger.info("Processing truth (GenPart)...")
        n_events = len(events)
        mx = self.config.max_objects

        truth = {}

        # Extract specific truth particles from GenPart by pdgId
        # statusFlags bit 13 = isLastCopy
        truth_types = {
            'electrons': 11,
            'muons':     13,
            'taus':      15,
        }

        for obj_name, abs_pdg in truth_types.items():
            d = {
                'pt':     np.zeros((n_events, mx), dtype=np.float32),
                'eta':    np.zeros((n_events, mx), dtype=np.float32),
                'phi':    np.zeros((n_events, mx), dtype=np.float32),
                'mass':   np.zeros((n_events, mx), dtype=np.float32),
                'pdgId':  np.zeros((n_events, mx), dtype=np.int32),
                'status': np.zeros((n_events, mx), dtype=np.int32),
                'n':      np.zeros(n_events, dtype=np.int32),
                'mask':   np.zeros((n_events, mx), dtype=bool),
            }

            for i, ev in enumerate(events):
                pt = _safe(ev, "GenPart_pt")
                if pt is None:
                    continue
                pt_np = ak.to_numpy(pt)
                pdg_np = ak.to_numpy(ev["GenPart_pdgId"])
                sf_np = ak.to_numpy(ev["GenPart_statusFlags"])

                # Select: correct pdgId AND isLastCopy (bit 13)
                sel = (np.abs(pdg_np) == abs_pdg) & ((sf_np & (1 << 13)) > 0)
                idx = np.where(sel)[0]
                n = min(len(idx), mx)
                if n == 0:
                    continue
                si = idx[np.argsort(pt_np[idx])[::-1][:n]]

                d['pt'][i, :n]     = pt_np[si]
                d['eta'][i, :n]    = ak.to_numpy(ev["GenPart_eta"])[si]
                d['phi'][i, :n]    = ak.to_numpy(ev["GenPart_phi"])[si]
                d['mass'][i, :n]   = ak.to_numpy(ev["GenPart_mass"])[si]
                d['pdgId'][i, :n]  = pdg_np[si]
                d['status'][i, :n] = ak.to_numpy(ev["GenPart_status"])[si]
                d['n'][i] = n
                d['mask'][i, :n] = True

            truth[obj_name] = d
            logger.info(f"  Truth {obj_name}: avg {d['n'].mean():.1f} per event")

        # Truth bosons (H=25, Z=23, W=24) from GenPart
        d = {
            'pt':     np.zeros((n_events, 5), dtype=np.float32),
            'eta':    np.zeros((n_events, 5), dtype=np.float32),
            'phi':    np.zeros((n_events, 5), dtype=np.float32),
            'mass':   np.zeros((n_events, 5), dtype=np.float32),
            'pdgId':  np.zeros((n_events, 5), dtype=np.int32),
            'status': np.zeros((n_events, 5), dtype=np.int32),
            'n':      np.zeros(n_events, dtype=np.int32),
            'mask':   np.zeros((n_events, 5), dtype=bool),
        }
        for i, ev in enumerate(events):
            pt = _safe(ev, "GenPart_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            pdg_np = ak.to_numpy(ev["GenPart_pdgId"])
            sf_np = ak.to_numpy(ev["GenPart_statusFlags"])
            sel = (np.isin(np.abs(pdg_np), [23, 24, 25])) & ((sf_np & (1 << 13)) > 0)
            idx = np.where(sel)[0]
            n = min(len(idx), 5)
            if n == 0:
                continue
            si = idx[np.argsort(pt_np[idx])[::-1][:n]]
            d['pt'][i, :n]     = pt_np[si]
            d['eta'][i, :n]    = ak.to_numpy(ev["GenPart_eta"])[si]
            d['phi'][i, :n]    = ak.to_numpy(ev["GenPart_phi"])[si]
            d['mass'][i, :n]   = ak.to_numpy(ev["GenPart_mass"])[si]
            d['pdgId'][i, :n]  = pdg_np[si]
            d['status'][i, :n] = ak.to_numpy(ev["GenPart_status"])[si]
            d['n'][i] = n
            d['mask'][i, :n] = True
        truth['bosons'] = d
        logger.info(f"  Truth bosons: avg {d['n'].mean():.1f} per event")

        # GenJets
        if "nGenJet" in events.fields:
            d = {
                'pt':   np.zeros((n_events, mx), dtype=np.float32),
                'eta':  np.zeros((n_events, mx), dtype=np.float32),
                'phi':  np.zeros((n_events, mx), dtype=np.float32),
                'mass': np.zeros((n_events, mx), dtype=np.float32),
                'n':    np.zeros(n_events, dtype=np.int32),
                'mask': np.zeros((n_events, mx), dtype=bool),
            }
            for i, ev in enumerate(events):
                pt = _safe(ev, "GenJet_pt")
                if pt is None:
                    continue
                pt_np = ak.to_numpy(pt)
                pm = pt_np > 20.0  # 20 GeV truth jet cut
                pt_np = pt_np[pm]
                n = min(len(pt_np), mx)
                if n == 0:
                    continue
                si = np.argsort(pt_np)[::-1][:n]
                d['pt'][i, :n]   = pt_np[si]
                d['eta'][i, :n]  = ak.to_numpy(_safe(ev, "GenJet_eta"))[pm][si]
                d['phi'][i, :n]  = ak.to_numpy(_safe(ev, "GenJet_phi"))[pm][si]
                m = _safe(ev, "GenJet_mass")
                if m is not None:
                    d['mass'][i, :n] = ak.to_numpy(m)[pm][si]
                d['n'][i] = n
                d['mask'][i, :n] = True
            truth['jets'] = d
            logger.info(f"  Truth jets: avg {d['n'].mean():.1f} per event")

        # GenMET
        if "GenMET_pt" in events.fields:
            truth['met'] = {
                'pt':  ak.to_numpy(events["GenMET_pt"]).astype(np.float32),
                'phi': ak.to_numpy(events["GenMET_phi"]).astype(np.float32),
            }
            logger.info(f"  Truth MET: avg {truth['met']['pt'].mean():.1f} GeV")

        return truth

    # ── HDF5 writer ───────────────────────────────────────────────────────

    def save_to_hdf5(self, data: Dict, output_file: str, input_file: str):
        logger.info(f"Writing: {output_file}")
        cfg = self.config
        n_events = len(data['event_info']['common']['experiment_id'])

        with h5py.File(output_file, 'w') as f:

            # /common
            grp_c = f.create_group('common')
            for obj in ['electrons', 'muons', 'photons', 'taus', 'jets', 'tracks']:
                sub = grp_c.create_group(obj)
                for key, arr in data[obj]['common'].items():
                    sub.create_dataset(key, data=arr, compression='gzip', compression_opts=4)

            grp_met = grp_c.create_group('met')
            for key, arr in data['met'].items():
                grp_met.create_dataset(key, data=arr, compression='gzip')

            grp_ev = grp_c.create_group('event')
            for key, arr in data['event_info']['common'].items():
                grp_ev.create_dataset(key, data=arr, compression='gzip')

            # /cms (experiment-specific)
            grp_x = f.create_group('cms')
            for obj in ['electrons', 'muons', 'photons', 'taus', 'jets']:
                if data[obj].get('cms'):
                    sub = grp_x.create_group(obj)
                    for key, arr in data[obj]['cms'].items():
                        sub.create_dataset(key, data=arr, compression='gzip', compression_opts=4)

            if data.get('pfcands') and data['pfcands'].get('cms'):
                sub = grp_x.create_group('pfcands')
                for key, arr in data['pfcands']['cms'].items():
                    sub.create_dataset(key, data=arr, compression='gzip', compression_opts=4)

            grp_xev = grp_x.create_group('event')
            for key, arr in data['event_info']['cms'].items():
                grp_xev.create_dataset(key, data=arr, compression='gzip')

            # /truth
            if 'truth' in data and data['truth']:
                grp_t = f.create_group('truth')
                for obj_name, obj_dict in data['truth'].items():
                    if obj_name == 'met':
                        sub = grp_t.create_group('met')
                        for key, arr in obj_dict.items():
                            sub.create_dataset(key, data=arr, compression='gzip')
                    else:
                        sub = grp_t.create_group(obj_name)
                        for key, arr in obj_dict.items():
                            sub.create_dataset(key, data=arr, compression='gzip',
                                               compression_opts=4)
                logger.info(f"  Truth groups: {list(data['truth'].keys())}")

            # /metadata
            meta = f.create_group('metadata')
            meta.attrs['experiment']          = 'CMS'
            meta.attrs['experiment_id']       = EXPERIMENT_ID
            meta.attrs['converter_version']   = CONVERTER_VERSION
            meta.attrs['converter_changelog'] = CONVERTER_CHANGELOG.get(CONVERTER_VERSION, '')
            meta.attrs['input_file']          = str(Path(input_file).name)
            meta.attrs['n_events']            = n_events
            has_pfcands = data.get('pfcands') is not None
            meta.attrs['input_format']        = 'PFNanoAOD' if has_pfcands else 'NanoAODv9'
            meta.attrs['has_pfcands']         = has_pfcands
            meta.attrs['sqrt_s_tev']          = 13.0
            meta.attrs['license']             = 'CC0-1.0'

            # Selection cuts
            meta.attrs['electron_pt_cut_gev'] = cfg.electron_pt_cut
            meta.attrs['electron_eta_cut']    = cfg.electron_eta_cut
            meta.attrs['electron_require_mva_loose'] = cfg.electron_require_mva_loose
            meta.attrs['muon_pt_cut_gev']     = cfg.muon_pt_cut
            meta.attrs['muon_eta_cut']        = cfg.muon_eta_cut
            meta.attrs['muon_require_loose']  = cfg.muon_require_loose
            meta.attrs['photon_pt_cut_gev']   = cfg.photon_pt_cut
            meta.attrs['tau_pt_cut_gev']      = cfg.tau_pt_cut
            meta.attrs['jet_pt_cut_gev']      = cfg.jet_pt_cut
            meta.attrs['jet_eta_cut']         = cfg.jet_eta_cut
            meta.attrs['jet_min_id']          = cfg.jet_min_id
            meta.attrs['pfcand_pt_cut_gev']   = cfg.pfcand_pt_cut
            meta.attrs['pfcand_pt_cut_all_gev'] = cfg.pfcand_pt_cut_all

        logger.info(f"  Written {n_events} events to {output_file}")

    def convert(self, input_file: str, output_file: str):
        data = self.process_file(input_file)
        self.save_to_hdf5(data, output_file, input_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert CMS NanoAODv9 ROOT files to HDF5 (TREASURE format)')
    parser.add_argument('input_files', nargs='+', help='Input NanoAOD ROOT files')
    parser.add_argument('-o', '--output', required=True, help='Output HDF5 file')
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--electron-pt-cut', type=float, default=7.0)
    parser.add_argument('--muon-pt-cut', type=float, default=5.0)
    parser.add_argument('--jet-pt-cut', type=float, default=30.0)
    args = parser.parse_args()

    cfg = CMSObjectConfig(
        electron_pt_cut=args.electron_pt_cut,
        muon_pt_cut=args.muon_pt_cut,
        jet_pt_cut=args.jet_pt_cut,
    )
    converter = CMSNanoAODConverter(cfg)

    if len(args.input_files) == 1:
        converter.convert(args.input_files[0], args.output)
    else:
        # Multiple files: process sequentially, concatenate
        all_data = None
        total_events = 0
        for fp in args.input_files:
            data = converter.process_file(fp)
            n = len(data['event_info']['common']['experiment_id'])
            if args.max_events and total_events + n > args.max_events:
                n = args.max_events - total_events
                # Truncate all arrays
                for obj_key in data:
                    if isinstance(data[obj_key], dict):
                        for sub_key in data[obj_key]:
                            if isinstance(data[obj_key][sub_key], dict):
                                for k, v in data[obj_key][sub_key].items():
                                    if isinstance(v, np.ndarray):
                                        data[obj_key][sub_key][k] = v[:n]
                            elif isinstance(data[obj_key][sub_key], np.ndarray):
                                data[obj_key][sub_key] = data[obj_key][sub_key][:n]

            if all_data is None:
                all_data = data
            else:
                # Concatenate
                for obj_key in all_data:
                    if isinstance(all_data[obj_key], dict):
                        for sub_key in all_data[obj_key]:
                            if isinstance(all_data[obj_key][sub_key], dict):
                                for k in all_data[obj_key][sub_key]:
                                    if isinstance(all_data[obj_key][sub_key][k], np.ndarray):
                                        all_data[obj_key][sub_key][k] = np.concatenate([
                                            all_data[obj_key][sub_key][k],
                                            data[obj_key][sub_key][k]
                                        ], axis=0)
                            elif isinstance(all_data[obj_key][sub_key], np.ndarray):
                                all_data[obj_key][sub_key] = np.concatenate([
                                    all_data[obj_key][sub_key],
                                    data[obj_key][sub_key]
                                ], axis=0)

            total_events += n
            if args.max_events and total_events >= args.max_events:
                break

        converter.save_to_hdf5(all_data, args.output, ','.join(args.input_files))


if __name__ == '__main__':
    main()
