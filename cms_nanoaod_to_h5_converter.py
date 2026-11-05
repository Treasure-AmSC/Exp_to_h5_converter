#!/usr/bin/env python3
"""
CMS NanoAOD → HDF5 Converter  (v2.1 — cross-experiment edition)
================================================================

Converts CMS Run 2 Open Data (NanoAODv9 or PFNanoAODv1) into padded HDF5
with the object fields and common selections used in the CMS study.

HDF5 Layout
-----------
  /common/    — Common object variables and experiment metadata.
                All energies/momenta in GeV, angles in radians.
  /cms/       — CMS-specific variables (MVA ID scores, DeepJet, etc.).
  /metadata   — Converter version, dataset info, selection cuts.

Schema
------
  common/electrons : pt, eta, phi, charge, mask, n
  common/muons     : pt, eta, phi, charge, mask, n
  common/taus      : pt, eta, phi, charge, mask, n
  common/photons   : pt, eta, phi, mask, n
  common/jets      : pt, eta, phi, mass, mask, n
  common/event     : experiment_id, is_simulation

  cms/electrons    : mvaIso, mvaFall17V2noIso_WPL, cutBased,
                     pfRelIso03_all, miniPFRelIso_chg
  cms/muons        : looseId, mediumId, tightId, pfRelIso03_all, miniPFRelIso_chg
  cms/taus         : idDeepTau2017v2p1VSjet, idDeepTau2017v2p1VSe,
                     idDeepTau2017v2p1VSmu, decayMode, rawDeepTau2017v2p1VSjet
  cms/photons      : cutBased, mvaID, electronVeto, pfRelIso03_all, pfRelIso03_chg
  cms/jets         : btagDeepFlavB, btagDeepFlavCvL, btagDeepFlavCvB, btagDeepFlavQG,
                     jetId, nConstituents
  cms/event        : event, run, luminosityBlock and available generator weights

Object Selection
----------------
  Electrons:  pT > 10 GeV, |eta| < 2.47, mvaFall17V2noIso_WPL
  Muons:      pT > 10 GeV, |eta| < 2.5, looseId
  Photons:    pT > 15 GeV, |eta| < 2.37, cutBased >= 1, electron veto, eta gap
  Taus:       pT > 20 GeV, |eta| < 2.5, VSjet bit 8, unit charge, eta gap
  Jets:       pT > 20 GeV, |eta| < 4.5, jetId bit 2

  The MC study's event-cleaning flags are applied to simulated inputs.
  PF candidate branches may be present but are not read or exported.

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
    electron_pt_cut: float = 10.0     # GeV
    electron_eta_cut: float = 2.47
    electron_require_mva_loose: bool = True  # mvaFall17V2noIso_WPL

    muon_pt_cut: float = 10.0         # GeV
    muon_eta_cut: float = 2.5
    muon_require_loose: bool = True    # looseId

    photon_pt_cut: float = 15.0       # GeV
    photon_eta_cut: float = 2.37
    photon_min_cutbased: int = 1      # >= 1 = loose

    tau_pt_cut: float = 20.0          # GeV
    tau_eta_cut: float = 2.5
    tau_min_deeptau_vsjet: int = 8    # require the loose VSjet bit

    jet_pt_cut: float = 20.0          # GeV
    jet_eta_cut: float = 4.5
    jet_min_id: int = 2               # require the tight ID bit

    pfcand_pt_cut: float = 0.5        # GeV (charged PFCands for common/tracks)
    pfcand_pt_cut_all: float = 0.2    # GeV (all PFCands for cms/pfcands)

    max_objects: int = 20
    max_tracks: int = 50               # charged PFCands → common/tracks
    max_pfcands: int = 500             # all PFCands → cms/pfcands


# ── Branch lists ──────────────────────────────────────────────────────────────

ELECTRON_BRANCHES = [
    "nElectron",
    "Electron_pt", "Electron_eta", "Electron_phi",
    "Electron_charge",
    "Electron_pfRelIso03_all", "Electron_miniPFRelIso_chg",
    "Electron_mvaFall17V2Iso", "Electron_mvaFall17V2noIso_WPL",
    "Electron_cutBased",
]

MUON_BRANCHES = [
    "nMuon",
    "Muon_pt", "Muon_eta", "Muon_phi",
    "Muon_charge",
    "Muon_looseId", "Muon_mediumId", "Muon_tightId",
    "Muon_pfRelIso03_all", "Muon_miniPFRelIso_chg",
]

PHOTON_BRANCHES = [
    "nPhoton",
    "Photon_pt", "Photon_eta", "Photon_phi",
    "Photon_pfRelIso03_all", "Photon_pfRelIso03_chg",
    "Photon_cutBased", "Photon_mvaID", "Photon_electronVeto",
]

TAU_BRANCHES = [
    "nTau",
    "Tau_pt", "Tau_eta", "Tau_phi",
    "Tau_charge", "Tau_decayMode",
    "Tau_idDeepTau2017v2p1VSjet", "Tau_idDeepTau2017v2p1VSe",
    "Tau_idDeepTau2017v2p1VSmu",
    "Tau_rawDeepTau2017v2p1VSjet",
]

JET_BRANCHES = [
    "nJet",
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_jetId", "Jet_nConstituents",
    "Jet_btagDeepFlavB", "Jet_btagDeepFlavCvL", "Jet_btagDeepFlavCvB", "Jet_btagDeepFlavQG",
]

EVENT_BRANCHES = ["event", "run", "luminosityBlock"]
MC_EVENT_FILTERS = [
    "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter",
    "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter", "Flag_hfNoisyHitsFilter",
]

# Keep the original signed MC weights. Confirm w/ CMS.
WEIGHT_BRANCHES = ["genWeight", "Generator_weight", "LHEWeight_originalXWGTUP"]

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


def _export_generator_weights(events):
    """Keep the available generator weights, including their signs."""
    native = {}
    for branch in WEIGHT_BRANCHES:
        if branch in events.fields:
            values = ak.to_numpy(events[branch]).astype(np.float64)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"Invalid generator weights: {branch}")
            native[branch] = values
    chosen = next((b for b in ("genWeight", "Generator_weight") if b in native), None)
    if chosen is not None:
        native["generator_weight"] = native[chosen].copy()
    return native


# ── Converter ─────────────────────────────────────────────────────────────────

class CMSNanoAODConverter:

    def __init__(self, config: Optional[CMSObjectConfig] = None):
        self.config = config or CMSObjectConfig()

    def process_file(self, input_file: str) -> Dict:
        """Read a NanoAOD ROOT file and return structured data dict."""
        logger.info(f"Reading: {input_file}")

        tree = uproot.open(f"{input_file}:Events")

        available = set(tree.keys())
        is_mc = bool(available.intersection(("nGenPart", "Pileup_nTrueInt", *WEIGHT_BRANCHES)))
        all_branches = [*ELECTRON_BRANCHES, *MUON_BRANCHES, *PHOTON_BRANCHES,
                        *TAU_BRANCHES, *JET_BRANCHES, *EVENT_BRANCHES]
        if is_mc:
            all_branches.extend(MC_EVENT_FILTERS)
        missing = sorted(set(all_branches) - available)
        if missing:
            raise ValueError(f"{input_file}: missing CMS study branches: {', '.join(missing)}")
        all_branches.extend(_get_available(tree, WEIGHT_BRANCHES))
        events = tree.arrays(all_branches, library="ak")
        n = len(events)
        logger.info(f"  {n} events loaded")

        if is_mc:
            event_keep = np.ones(n, dtype=bool)
            for flag in MC_EVENT_FILTERS:
                event_keep &= ak.to_numpy(events[flag]) == 1
            events = events[event_keep]

        result = {
            'electrons': self._process_electrons(events),
            'muons':     self._process_muons(events),
            'photons':   self._process_photons(events),
            'taus':      self._process_taus(events),
            'jets':      self._process_jets(events),
            # Only retain PFNano input detection for the old metadata.
            'pfcands':   {} if "nPFCands" in available else None,
            'event_info': self._process_event_info(events, is_mc),
        }

        # Save event weights alongside the other CMS event information.
        result['event_info']['cms'].update(_export_generator_weights(events))

        return result

    # ── Object processors ─────────────────────────────────────────────────

    def _process_electrons(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'charge']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['mvaIso', 'mvaFall17V2noIso_WPL', 'cutBased',
                         'pfRelIso03_all', 'miniPFRelIso_chg']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Electron_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Electron_eta"])

            # Selection
            pm = np.isfinite(pt_np) & (pt_np > self.config.electron_pt_cut) & (np.abs(eta_np) < self.config.electron_eta_cut)
            if self.config.electron_require_mva_loose:
                wpl = _safe(ev, "Electron_mvaFall17V2noIso_WPL")
                if wpl is not None:
                    pm = pm & (ak.to_numpy(wpl) == 1)

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
            c['n'][i] = n
            c['mask'][i, :n] = True

            # CMS-specific
            for key, branch in [('mvaIso', 'Electron_mvaFall17V2Iso'),
                                ('mvaFall17V2noIso_WPL', 'Electron_mvaFall17V2noIso_WPL'),
                                ('cutBased', 'Electron_cutBased'),
                                ('pfRelIso03_all', 'Electron_pfRelIso03_all'),
                                ('miniPFRelIso_chg', 'Electron_miniPFRelIso_chg')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_muons(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'charge']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['looseId', 'mediumId', 'tightId',
                         'pfRelIso03_all', 'miniPFRelIso_chg']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Muon_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Muon_eta"])

            pm = np.isfinite(pt_np) & (pt_np > self.config.muon_pt_cut) & (np.abs(eta_np) < self.config.muon_eta_cut)
            if self.config.muon_require_loose:
                lid = _safe(ev, "Muon_looseId")
                if lid is not None:
                    pm = pm & (ak.to_numpy(lid) == 1)

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
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('looseId', 'Muon_looseId'),
                                ('mediumId', 'Muon_mediumId'),
                                ('tightId', 'Muon_tightId'),
                                ('pfRelIso03_all', 'Muon_pfRelIso03_all'),
                                ('miniPFRelIso_chg', 'Muon_miniPFRelIso_chg')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_photons(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['cutBased', 'mvaID', 'electronVeto', 'pfRelIso03_all', 'pfRelIso03_chg']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Photon_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Photon_eta"])

            abs_eta = np.abs(eta_np)
            pm = np.isfinite(pt_np) & (pt_np > self.config.photon_pt_cut) & (abs_eta < self.config.photon_eta_cut)
            pm = pm & ((abs_eta < 1.37) | (abs_eta > 1.52))
            pm = pm & (ak.to_numpy(ev["Photon_electronVeto"]) == 1)
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
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('cutBased', 'Photon_cutBased'),
                                ('mvaID', 'Photon_mvaID'),
                                ('electronVeto', 'Photon_electronVeto'),
                                ('pfRelIso03_chg', 'Photon_pfRelIso03_chg'),
                                ('pfRelIso03_all', 'Photon_pfRelIso03_all')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_taus(self, events) -> Dict:
        n_events = len(events)
        mx = self.config.max_objects
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt', 'eta', 'phi', 'charge']}
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

            abs_eta = np.abs(eta_np)
            pm = np.isfinite(pt_np) & (pt_np > self.config.tau_pt_cut) & (abs_eta < self.config.tau_eta_cut)
            pm = pm & ~((abs_eta > 1.37) & (abs_eta < 1.52))
            pm = pm & (np.abs(ak.to_numpy(ev["Tau_charge"])) == 1)
            if self.config.tau_min_deeptau_vsjet > 0:
                dt = _safe(ev, "Tau_idDeepTau2017v2p1VSjet")
                if dt is not None:
                    pm = pm & ((ak.to_numpy(dt) & self.config.tau_min_deeptau_vsjet) == self.config.tau_min_deeptau_vsjet)

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
             for k in ['pt', 'eta', 'phi', 'mass']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        cms = {k: np.zeros((n_events, mx), dtype=np.float32)
               for k in ['btagDeepFlavB', 'btagDeepFlavCvL', 'btagDeepFlavCvB',
                         'btagDeepFlavQG', 'jetId', 'nConstituents']}

        for i, ev in enumerate(events):
            pt = _safe(ev, "Jet_pt")
            if pt is None:
                continue
            pt_np = ak.to_numpy(pt)
            eta_np = ak.to_numpy(ev["Jet_eta"])

            pm = np.isfinite(pt_np) & (pt_np > self.config.jet_pt_cut) & (np.abs(eta_np) < self.config.jet_eta_cut)
            if self.config.jet_min_id > 0:
                jid = _safe(ev, "Jet_jetId")
                if jid is not None:
                    pm = pm & ((ak.to_numpy(jid) & self.config.jet_min_id) == self.config.jet_min_id)

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
            c['n'][i] = n
            c['mask'][i, :n] = True

            for key, branch in [('btagDeepFlavB', 'Jet_btagDeepFlavB'),
                                ('btagDeepFlavCvL', 'Jet_btagDeepFlavCvL'),
                                ('btagDeepFlavCvB', 'Jet_btagDeepFlavCvB'),
                                ('btagDeepFlavQG', 'Jet_btagDeepFlavQG'),
                                ('jetId', 'Jet_jetId'),
                                ('nConstituents', 'Jet_nConstituents')]:
                v = _safe(ev, branch)
                if v is not None:
                    cms[key][i, :n] = ak.to_numpy(v)[si]

        return {'common': c, 'cms': cms}

    def _process_event_info(self, events, is_mc) -> Dict:
        n_events = len(events)
        c = {
            'experiment_id': np.ones(n_events, dtype=np.int32) * EXPERIMENT_ID,
            'is_simulation': np.full(n_events, is_mc, dtype=np.int32),
        }
        cms_ev = {
            'event':            np.zeros(n_events, dtype=np.int64),
            'run':              np.zeros(n_events, dtype=np.int32),
            'luminosityBlock':  np.zeros(n_events, dtype=np.int32),
        }

        # Event IDs
        for key, branch in [('event', 'event'), ('run', 'run'),
                            ('luminosityBlock', 'luminosityBlock')]:
            v = _safe(events, branch)
            if v is not None:
                cms_ev[key] = ak.to_numpy(v)

        return {'common': c, 'cms': cms_ev}

    # ── HDF5 writer ───────────────────────────────────────────────────────

    def save_to_hdf5(self, data: Dict, output_file: str, input_file: str):
        logger.info(f"Writing: {output_file}")
        cfg = self.config
        n_events = len(data['event_info']['common']['experiment_id'])

        with h5py.File(output_file, 'w') as f:

            # /common
            grp_c = f.create_group('common')
            for obj in ['electrons', 'muons', 'photons', 'taus', 'jets']:
                sub = grp_c.create_group(obj)
                for key, arr in data[obj]['common'].items():
                    sub.create_dataset(key, data=arr, compression='gzip', compression_opts=4)

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

            grp_xev = grp_x.create_group('event')
            for key, arr in data['event_info']['cms'].items():
                grp_xev.create_dataset(key, data=arr, compression='gzip')

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

            # Sum over the events written to this H5, with no luminosity normalization.
            weights = data['event_info']['cms'].get('generator_weight')
            if weights is not None:
                meta.attrs['written_genweight_sum'] = float(np.sum(weights, dtype=np.float64))
                meta.attrs['written_genweight_sum2'] = float(np.sum(np.square(weights), dtype=np.float64))

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
    parser.add_argument('--electron-pt-cut', type=float, default=10.0)
    parser.add_argument('--muon-pt-cut', type=float, default=10.0)
    parser.add_argument('--jet-pt-cut', type=float, default=20.0)
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
