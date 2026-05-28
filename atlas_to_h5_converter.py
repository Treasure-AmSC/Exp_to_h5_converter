#!/usr/bin/env python3
"""
ATLAS DAOD_PHYSLITE → HDF5 Converter  (v2.1 — cross-experiment edition)
========================================================================

Converts ATLAS Run 2 Open Data (PHYSLITE format) into analysis-ready HDF5
files for the TREASURE foundation model pipeline.

HDF5 Layout
-----------
  /common/    — Cross-experiment variables (same keys for ATLAS and CMS).
                All energies/momenta in GeV, angles in radians.
  /atlas/     — ATLAS-specific variables (ID flags, b-tagging, etc.).
  /truth/     — MC truth particles (electrons, muons, photons, taus,
                bosons, jets, MET). Only present for simulation.
  /metadata   — HDF5 group attributes: experiment, DSID, process name,
                cross-section, selection cuts, provenance.

Schema
------
  common/electrons : pt, eta, phi, charge, trk_iso03, mask, n
  common/muons     : pt, eta, phi, charge, trk_iso03, mask, n
  common/taus      : pt, eta, phi, charge, is_1prong, mask, n
  common/photons   : pt, eta, phi, trk_iso03, mask, n
  common/jets      : pt, eta, phi, mass, n_trk, mask, n
  common/tracks    : pt, eta, phi, d0, z0, mask, n
  common/met       : pt, phi, sumet                    (scalar per event)
  common/event     : pvx, pvy, pvz, mu, experiment_id, is_simulation

  atlas/electrons  : LHLoose, LHMedium, LHTight, topoetcone20, ptvarcone30, mass
  atlas/muons      : quality, muonType, passIDCuts, passPresel,
                     topoetcone20, ptvarcone30
  atlas/taus       : NNDecayMode, RNNJetScore, RNNEleScore,
                     EleRNNLoose/Medium/Tight_v1
  atlas/photons    : isLoose, isTight, isCleaning, author,
                     topoetcone20, topoetcone40, ptcone20
  atlas/jets       : DL1d_pb/pc/pu, GN2_pb/pc/pu, QG_nTracks/Width/C1
  atlas/tracks     : qOverP, chiSquared, nDoF
  atlas/event      : event_number, run_number, mcChannelNumber

  truth/electrons  : pt, eta, phi, mass, pdgId, status, n, mask
  truth/muons      : pt, eta, phi, mass, pdgId, status, n, mask
  truth/photons    : pt, eta, phi, mass, pdgId, status, n, mask
  truth/taus       : pt, eta, phi, mass, pdgId, status, n, mask
  truth/bosons     : pt, eta, phi, mass, pdgId, status, n, mask  (H, Z, W)
  truth/jets       : pt, eta, phi, mass, n, mask   (AntiKt4TruthDressedWZ)
  truth/met        : pt, phi, sumet

Object Selection (mimicking H→ZZ→4ℓ analysis)
----------------------------------------------
  Electrons:
    - pT > 5 GeV
    - LH Loose ID (DFCommonElectronsLHLoose)
    - Author == 1 (single-track) or 16 (forward)
    - Object Quality: (OQ & 1446) == 0
    - |η| < 2.47
    - |z0·sinθ| < 3m mm

  Muons:
    - pT > 5 GeV  (calo-tagged: pT > 15 GeV)
    - DFCommonMuonPassPreselection (standard ATLAS muon preselection)
    - |η| < 2.7
    - Combined + CaloTagged: |d0| < 3 mm, |z0·sinθ| < 3 mm
    - StandAlone: no impact parameter cut

  Photons:
    - pT > 10 GeV
    - DFCommonPhotonsIsEMLoose

  Taus:
    - pT > 20 GeV
    - RNNJetScore > 0.01

  Jets:
    - pT > 30 GeV
    - |η| < 4.5
    - DFCommonJets_jetClean_LooseBad (standard ATLAS Loose jet cleaning)

  Tracks:
    - pT > 500 MeV
    - |η| < 2.5
    - |d0| < 2 mm
    - |z0·sinθ| < 3 mm

  All cuts are configurable via PhysicsObjectConfig.

Isolation (common/*/trk_iso03)
------------------------------
  I_trk = sum_pT(tracks, ΔR < 0.3, pT > 0.5 GeV, |z0·sinθ| < 3 mm) / pT
  Computed per-object from InDetTrackParticles.

Metadata (/metadata attrs)
--------------------------
  converter_version, converter_changelog, experiment, experiment_id,
  input_file, n_events, DSID, process name, generator, cross-section (pb),
  filter efficiency, k-factor, √s, data-taking year, MC campaign,
  license, DOI, citation.  All selection cuts recorded for reproducibility.
  Uses atlasopenmagic package if installed, else hardcoded lookup table.

References
----------
  - ATLAS Open Data: https://opendata.atlas.cern
  - PHYSLITE format: Eur. Phys. J. C 82 (2022) 105
  - Jet cleaning: Eur. Phys. J. C 81 (2021) 689 [arXiv:2007.02645]
  - Muon selection: Eur. Phys. J. C 81 (2021) 578 [arXiv:2012.00578]
  - Electron ID: Eur. Phys. J. C 79 (2019) 639 [arXiv:1902.04655]
"""

import numpy as np
import h5py
import uproot
import awkward as ak
import vector
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONVERTER_VERSION = "2.1.0"
EXPERIMENT_ID  = 0          # 0 = ATLAS, 1 = CMS
CONVERTER_CHANGELOG = {
    "2.1.0": "Add truth particles, CERN Open Data metadata, HZZ-style quality cuts, "
             "jet LooseBad cleaning, muon passPresel, electron OQ/author/z0sinθ, "
             "production versioning",
    "2.0.0": "Cross-experiment format (common/atlas split), charge for e/μ/τ, "
             "trk_iso03, muonType, NNDecayMode, is_1prong",
    "1.0.0": "Initial format: flat H5, MeV units, no quality cuts",
}

# ── CERN Open Data metadata ──────────────────────────────────────────────────

# Known MC samples from ATLAS Open Data PHYSLITE release (mc20a, 13 TeV)
# Source: https://opendata.atlas.cern/docs/data/for_research/metadata
# Cross sections from atlasopenmagic / AMI; filter_efficiency × k_factor included
MC_METADATA = {
    # Higgs signals
    345060: {'process': 'ggH125 → ZZ → 4ℓ', 'generator': 'PowhegPythia8EvtGen',
             'cross_section_pb': 0.01297, 'filter_efficiency': 1.0, 'k_factor': 1.1,
             'keywords': 'Higgs,ZZ,4lepton,SM'},
    344235: {'process': 'VBFH125 → ZZ → 4ℓ', 'generator': 'PowhegPythia8EvtGen',
             'cross_section_pb': 0.001012, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'Higgs,VBF,ZZ,4lepton,SM'},
    341488: {'process': 'ggH125 → γγ', 'generator': 'PowhegPythia8',
             'cross_section_pb': 0.1057, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'Higgs,diphoton,SM'},
    # ZZ continuum background
    363356: {'process': 'qq → ZZ → 4ℓ (Sherpa)', 'generator': 'Sherpa_222',
             'cross_section_pb': 1.2523, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'diboson,ZZ,4lepton,SM'},
    364250: {'process': 'qq → ZZ → ℓℓνν (Sherpa)', 'generator': 'Sherpa_222',
             'cross_section_pb': 4.582, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'diboson,ZZ,SM'},
    # Diphoton backgrounds
    364350: {'process': 'γγ myy 0-50 (Sherpa)', 'generator': 'Sherpa_224',
             'cross_section_pb': 87700.0, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'diphoton,SM'},
    364351: {'process': 'γγ myy 50-100 (Sherpa)', 'generator': 'Sherpa_224',
             'cross_section_pb': 1476.0, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'diphoton,SM'},
    # Top backgrounds
    410470: {'process': 'ttbar (non-allhad)', 'generator': 'PhPy8EG_A14',
             'cross_section_pb': 729.77, 'filter_efficiency': 0.543, 'k_factor': 1.1396,
             'keywords': 'top,ttbar,SM'},
    410472: {'process': 'ttbar (dilepton)', 'generator': 'PhPy8EG_A14',
             'cross_section_pb': 729.77, 'filter_efficiency': 0.105, 'k_factor': 1.1396,
             'keywords': 'top,ttbar,dilepton,SM'},
    # W/Z + jets
    364156: {'process': 'Z→ee (Sherpa)', 'generator': 'Sherpa_221',
             'cross_section_pb': 1950.63, 'filter_efficiency': 1.0, 'k_factor': 0.9751,
             'keywords': 'Zjets,Zee,SM'},
    364170: {'process': 'Z→μμ (Sherpa)', 'generator': 'Sherpa_221',
             'cross_section_pb': 1950.63, 'filter_efficiency': 1.0, 'k_factor': 0.9751,
             'keywords': 'Zjets,Zmumu,SM'},
    # WZ
    363357: {'process': 'WZ → ℓℓℓν (Sherpa)', 'generator': 'Sherpa_222',
             'cross_section_pb': 4.5877, 'filter_efficiency': 1.0, 'k_factor': 1.0,
             'keywords': 'diboson,WZ,SM'},
}


def _extract_dsid(filename: str) -> Optional[int]:
    """Extract the ATLAS dataset ID (DSID) from a PHYSLITE filename.

    Patterns:
      DAOD_PHYSLITE.37620644._000012.pool.root.1  (rucio file ID, not DSID)
      mc20_13TeV.345060.PowhegPythia8...           (DSID = 345060)

    For rucio files, the mcChannelNumber in the events IS the DSID.
    We try to extract from the filename first; if not found, return None
    (caller should use mcChannelNumber from the event data).
    """
    import re
    # Try mc20 naming: mc20_13TeV.DSID.Generator...
    m = re.search(r'mc20_13TeV\.(\d{6})\.', filename)
    if m:
        return int(m.group(1))
    # Try DAOD_PHYSLITE.XXXXXXXX pattern — this is a rucio file ID, NOT a DSID
    # Return None and let the caller use mcChannelNumber
    return None


def _lookup_mc_metadata(dsid: int) -> dict:
    """Look up MC metadata by DSID.
    First tries atlasopenmagic package, falls back to hardcoded table."""
    if dsid is None or dsid == 0:
        return {}

    # Try atlasopenmagic first
    try:
        import atlasopenmagic as aom
        aom.set_release('2024r-pp')
        info = aom.get_metadata(str(dsid))
        if info:
            return {
                'process': info.get('physics_short', ''),
                'generator': info.get('generators', ''),
                'cross_section_pb': float(info.get('cross_section', 0)),
                'filter_efficiency': float(info.get('filter_efficiency', 1.0)),
                'k_factor': float(info.get('k_factor', 1.0)),
                'keywords': info.get('keywords', ''),
                'is_mc': True,
            }
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"atlasopenmagic lookup failed for DSID {dsid}: {e}")

    # Fallback to hardcoded table
    if dsid in MC_METADATA:
        return {**MC_METADATA[dsid], 'is_mc': True}

    # Unknown DSID
    return {'is_mc': True, 'process': f'unknown (DSID {dsid})'}

# ── Isolation parameters (must match CMS converter) ──────────────────────────
ISO_CONE         = 0.3      # ΔR
ISO_PT_FLOOR_GEV = 0.5      # track pT floor in GeV
ISO_Z0ST_CUT_MM  = 3.0      # |z0·sinθ| cut in mm

MeV = 1e-3                  # convert MeV → GeV


@dataclass
class PhysicsObjectConfig:
    electron_pt_cut: float = 5_000   # MeV
    muon_pt_cut:     float = 5_000   # MeV
    photon_pt_cut:   float = 10_000  # MeV
    tau_pt_cut:      float = 20_000  # MeV
    jet_pt_cut:      float = 20_000  # MeV 
    track_pt_cut:    float = 500     # MeV  (= ISO_PT_FLOOR so tracks used for iso are kept)
    max_objects:     int   = 50
    max_tracks:      int   = 50

    # ── Quality selections (mimicking H→ZZ→4ℓ analysis) ────────────
    # Electrons: require LHLoose, author 1 or 16, |eta_cl| < 2.47,
    #            OQ clean, |z0*sinθ| < 0.5 mm
    electron_require_loose: bool = True
    electron_eta_cut: float = 2.47      # |eta_cluster|
    electron_z0sintheta_cut: float = 3  # mm
    electron_require_author: bool = True  # author == 1 or 16
    electron_require_oq: bool = True      # (OQ & 1446) == 0

    # Muons: use DFCommonMuonPassPreselection (standard ATLAS preselection),
    #        |eta| < 2.7, loose impact parameter cuts
    muon_max_quality: int = 99     # disabled — use passPresel instead
    muon_require_presel: bool = True  # DFCommonMuonPassPreselection
    muon_eta_cut: float = 2.7
    muon_d0_cut: float = 3.0           # mm (looser than HZZ's 1 mm)
    muon_z0sintheta_cut: float = 3.0   # mm (looser than HZZ's 0.5 mm)
    muon_calo_pt_cut: float = 15_000   # MeV — higher pT for calo-tagged muons

    # Photons: require isLoose
    photon_require_loose: bool = True

    # Taus: require RNNJetScore > loose WP (~0.01) — removes QCD fakes
    tau_rnn_jet_cut: float = 0.01  # set 0 to disable

    # Jets: require |eta| < 4.5, LooseBad cleaning, pT > 30 GeV (HZZ default)
    jet_eta_cut: float = 4.5
    jet_clean_loose: bool = True     # require DFCommonJets_jetClean_LooseBad

    # Tracks: require |eta| < 2.5, |d0| < 2 mm, |z0·sinθ| < 3 mm
    track_eta_cut: float = 2.5
    track_d0_cut: float = 2.0      # mm
    track_z0sintheta_cut: float = 3.0  # mm


# ── Branch lists ─────────────────────────────────────────────────────────────

ELECTRON_BRANCHES = [
    "AnalysisElectronsAuxDyn.pt",
    "AnalysisElectronsAuxDyn.eta",
    "AnalysisElectronsAuxDyn.phi",
    "AnalysisElectronsAuxDyn.m",
    "AnalysisElectronsAuxDyn.charge",
    "AnalysisElectronsAuxDyn.topoetcone20",
    "AnalysisElectronsAuxDyn.ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHLoose",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHMedium",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHTight",
    # HZZ quality cuts
    "AnalysisElectronsAuxDyn.author",
    "AnalysisElectronsAuxDyn.OQ",
    # Track info for z0*sinθ cut
    "AnalysisElectronsAuxDyn.z0",
    "AnalysisElectronsAuxDyn.z0sinTheta",
]

MUON_BRANCHES = [
    "AnalysisMuonsAuxDyn.pt",
    "AnalysisMuonsAuxDyn.eta",
    "AnalysisMuonsAuxDyn.phi",
    "AnalysisMuonsAuxDyn.charge",
    "AnalysisMuonsAuxDyn.quality",
    "AnalysisMuonsAuxDyn.muonType",
    "AnalysisMuonsAuxDyn.DFCommonMuonPassIDCuts",
    "AnalysisMuonsAuxDyn.DFCommonMuonPassPreselection",
    "AnalysisMuonsAuxDyn.ptvarcone30",
    "AnalysisMuonsAuxDyn.topoetcone20",
    # HZZ impact parameter cuts
    "AnalysisMuonsAuxDyn.d0",
    "AnalysisMuonsAuxDyn.z0sinTheta",
]

PHOTON_BRANCHES = [
    "AnalysisPhotonsAuxDyn.pt",
    "AnalysisPhotonsAuxDyn.eta",
    "AnalysisPhotonsAuxDyn.phi",
    "AnalysisPhotonsAuxDyn.m",
    "AnalysisPhotonsAuxDyn.topoetcone20",
    "AnalysisPhotonsAuxDyn.topoetcone40",
    "AnalysisPhotonsAuxDyn.ptcone20",
    "AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMLoose",
    # DFCommonPhotonsIsEMMedium does NOT exist in PHYSLITE (only Loose and Tight)
    "AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMTight",
    "AnalysisPhotonsAuxDyn.DFCommonPhotonsCleaning",
    "AnalysisPhotonsAuxDyn.author",
]

TAU_BRANCHES = [
    "AnalysisTauJetsAuxDyn.pt",
    "AnalysisTauJetsAuxDyn.eta",
    "AnalysisTauJetsAuxDyn.phi",
    "AnalysisTauJetsAuxDyn.charge",
    # NNDecayMode: 0/1/2 = 1-prong, 10/11 = 3-prong (same encoding as CMS Tau_decayMode)
    "AnalysisTauJetsAuxDyn.NNDecayMode",
    # Jet discriminant (Run-3 tagger)
    "AnalysisTauJetsAuxDyn.RNNJetScore",
    # Electron veto discriminant + working points
    "AnalysisTauJetsAuxDyn.RNNEleScore",
    "AnalysisTauJetsAuxDyn.EleRNNLoose_v1",
    "AnalysisTauJetsAuxDyn.EleRNNMedium_v1",
    "AnalysisTauJetsAuxDyn.EleRNNTight_v1",
]

JET_BRANCHES = [
    "AnalysisJetsAuxDyn.pt",
    "AnalysisJetsAuxDyn.eta",
    "AnalysisJetsAuxDyn.phi",
    "AnalysisJetsAuxDyn.m",
    "AnalysisJetsAuxDyn.NumTrkPt500",
    "AnalysisJetsAuxDyn.DFCommonJets_QGTagger_NTracks",
    "AnalysisJetsAuxDyn.DFCommonJets_QGTagger_TracksWidth",
    "AnalysisJetsAuxDyn.DFCommonJets_QGTagger_TracksC1",
    # Jet cleaning
    "AnalysisJetsAuxDyn.DFCommonJets_jetClean_LooseBad",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu",
    "BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pb",
    "BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pc",
    "BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pu",
]

TRACK_BRANCHES = [
    "InDetTrackParticlesAuxDyn.qOverP",
    "InDetTrackParticlesAuxDyn.theta",
    "InDetTrackParticlesAuxDyn.phi",
    "InDetTrackParticlesAuxDyn.d0",
    "InDetTrackParticlesAuxDyn.z0",
    "InDetTrackParticlesAuxDyn.chiSquared",
    "InDetTrackParticlesAuxDyn.numberDoF",
]

# ── Truth branches (MC only) ──────────────────────────────────────────────
TRUTH_ELECTRON_BRANCHES = [
    "TruthElectronsAuxDyn.pt",
    "TruthElectronsAuxDyn.eta",
    "TruthElectronsAuxDyn.phi",
    "TruthElectronsAuxDyn.m",
    "TruthElectronsAuxDyn.pdgId",
    "TruthElectronsAuxDyn.status",
    "TruthElectronsAuxDyn.barcode",
]
TRUTH_MUON_BRANCHES = [
    "TruthMuonsAuxDyn.pt",
    "TruthMuonsAuxDyn.eta",
    "TruthMuonsAuxDyn.phi",
    "TruthMuonsAuxDyn.m",
    "TruthMuonsAuxDyn.pdgId",
    "TruthMuonsAuxDyn.status",
    "TruthMuonsAuxDyn.barcode",
]
TRUTH_PHOTON_BRANCHES = [
    "TruthPhotonsAuxDyn.pt",
    "TruthPhotonsAuxDyn.eta",
    "TruthPhotonsAuxDyn.phi",
    "TruthPhotonsAuxDyn.m",
    "TruthPhotonsAuxDyn.pdgId",
    "TruthPhotonsAuxDyn.status",
    "TruthPhotonsAuxDyn.barcode",
]
TRUTH_TAU_BRANCHES = [
    "TruthTausAuxDyn.pt",
    "TruthTausAuxDyn.eta",
    "TruthTausAuxDyn.phi",
    "TruthTausAuxDyn.m",
    "TruthTausAuxDyn.pdgId",
    "TruthTausAuxDyn.status",
    "TruthTausAuxDyn.barcode",
]
TRUTH_BOSON_BRANCHES = [
    "TruthBosonAuxDyn.pt",
    "TruthBosonAuxDyn.eta",
    "TruthBosonAuxDyn.phi",
    "TruthBosonAuxDyn.m",
    "TruthBosonAuxDyn.pdgId",
    "TruthBosonAuxDyn.status",
    "TruthBosonAuxDyn.barcode",
]
TRUTH_JET_BRANCHES = [
    "AntiKt4TruthDressedWZJetsAuxDyn.pt",
    "AntiKt4TruthDressedWZJetsAuxDyn.eta",
    "AntiKt4TruthDressedWZJetsAuxDyn.phi",
    "AntiKt4TruthDressedWZJetsAuxDyn.m",
]
TRUTH_MET_BRANCHES = [
    "MET_TruthAuxDyn.mpx",
    "MET_TruthAuxDyn.mpy",
    "MET_TruthAuxDyn.sumet",
]

MET_BRANCHES = [
    "MET_Core_AnalysisMETAuxDyn.mpx",
    "MET_Core_AnalysisMETAuxDyn.mpy",
    "MET_Core_AnalysisMETAuxDyn.sumet",
]

EVENT_BRANCHES = [
    "EventInfoAuxDyn.eventNumber",
    "EventInfoAuxDyn.runNumber",
    "EventInfoAuxDyn.mcChannelNumber",
    "EventInfoAuxDyn.averageInteractionsPerCrossing",
]

VERTEX_BRANCHES = [
    "PrimaryVerticesAuxDyn.x",
    "PrimaryVerticesAuxDyn.y",
    "PrimaryVerticesAuxDyn.z",
    "PrimaryVerticesAuxDyn.vertexType",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_get(ev, name):
    return ev[name] if name in ev.fields else None


def _get_available(tree, branch_list):
    keys = set(tree.keys())
    avail = [b for b in branch_list if b in keys]
    missing = [b for b in branch_list if b not in keys]
    if missing:
        logger.debug(f"Branches not found: {missing}")
    return avail


def _compute_track_iso(
    obj_eta: np.ndarray,   # shape (N_obj,)
    obj_phi: np.ndarray,
    obj_pt_gev: np.ndarray,
    trk_eta: np.ndarray,   # shape (N_trk,)
    trk_phi: np.ndarray,
    trk_pt_gev: np.ndarray,
    trk_z0: np.ndarray,    # mm — z0 already w.r.t. PV in ATLAS track branches
    trk_theta: np.ndarray,
    cone: float = ISO_CONE,
    pt_floor: float = ISO_PT_FLOOR_GEV,
    z0st_cut: float = ISO_Z0ST_CUT_MM,
) -> np.ndarray:
    """
    Compute relative track isolation for a set of objects in one event.

    Returns array of shape (N_obj,) with I_trk = sum_pT / pT(object).
    Tracks matched to the object itself (ΔR < 0.01) are excluded.
    Objects with pT=0 get iso=0.
    """
    # Track quality: pT floor and longitudinal impact parameter cut
    z0st = np.abs(trk_z0 * np.sin(trk_theta))   # |z0·sinθ| in mm
    good = (trk_pt_gev > pt_floor) & (z0st < z0st_cut)
    trk_eta = trk_eta[good]
    trk_phi = trk_phi[good]
    trk_pt  = trk_pt_gev[good]

    iso = np.zeros(len(obj_eta), dtype=np.float32)

    for j, (oeta, ophi, opt) in enumerate(zip(obj_eta, obj_phi, obj_pt_gev)):
        if opt == 0:
            continue
        deta = trk_eta - oeta
        dphi = trk_phi - ophi
        dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
        dR   = np.sqrt(deta**2 + dphi**2)
        in_cone  = (dR < cone) & (dR > 0.01)   # exclude matched track
        iso[j] = float(np.sum(trk_pt[in_cone])) / opt

    return iso


# ── Main converter class ──────────────────────────────────────────────────────

class DAOD_PHYSLITE_Converter:

    def __init__(self, config: Optional[PhysicsObjectConfig] = None):
        self.config = config or PhysicsObjectConfig()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def read_root_file(self, filepath: str) -> Dict:
        logger.info(f"Reading: {filepath}")
        root_file = uproot.open(filepath)
        tree = root_file["CollectionTree"]

        all_branches = []
        for bl in [ELECTRON_BRANCHES, MUON_BRANCHES, PHOTON_BRANCHES,
                   TAU_BRANCHES, JET_BRANCHES, TRACK_BRANCHES,
                   MET_BRANCHES, EVENT_BRANCHES, VERTEX_BRANCHES]:
            all_branches.extend(_get_available(tree, bl))

        # Truth branches (MC only — silently skip if not found)
        truth_branch_lists = [
            TRUTH_ELECTRON_BRANCHES, TRUTH_MUON_BRANCHES,
            TRUTH_PHOTON_BRANCHES, TRUTH_TAU_BRANCHES,
            TRUTH_BOSON_BRANCHES, TRUTH_JET_BRANCHES,
            TRUTH_MET_BRANCHES,
        ]
        for bl in truth_branch_lists:
            all_branches.extend(_get_available(tree, bl))

        events = tree.arrays(all_branches, library="ak")
        n = len(events)
        logger.info(f"  {n} events loaded")

        # Pre-build per-event track arrays for isolation computation
        track_arrays = self._load_track_arrays(events)

        # Check if truth is available
        has_truth = "TruthElectronsAuxDyn.pt" in events.fields

        result = {
            'electrons':  self._process_electrons(events, track_arrays),
            'muons':      self._process_muons(events, track_arrays),
            'taus':       self._process_taus(events),
            'photons':    self._process_photons(events, track_arrays),
            'jets':       self._process_jets(events),
            'tracks':     self._process_tracks(events),
            'met':        self._process_met(events),
            'event_info': self._process_event_info(events),
        }

        if has_truth:
            result['truth'] = self._process_truth(events)

        return result

    # ── Track pre-loading (for isolation) ────────────────────────────────────

    def _load_track_arrays(self, events) -> List[Dict]:
        """
        Pre-compute per-event track kinematics in GeV for isolation.
        Returns list of dicts, one per event.
        """
        logger.info("Pre-loading tracks for isolation computation...")
        n_events = len(events)
        result = []

        for i, ev in enumerate(events):
            if "InDetTrackParticlesAuxDyn.qOverP" not in ev.fields:
                result.append(None)
                continue

            t_qop   = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.qOverP"])
            t_theta = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.theta"])
            t_phi   = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.phi"])
            t_d0    = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.d0"])    \
                      if "InDetTrackParticlesAuxDyn.d0" in ev.fields else np.zeros_like(t_qop)
            t_z0    = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.z0"])    \
                      if "InDetTrackParticlesAuxDyn.z0" in ev.fields else np.zeros_like(t_qop)

            abs_qop = np.abs(t_qop) + 1e-12
            sin_th  = np.sin(t_theta)
            t_pt_mev = np.abs(sin_th / abs_qop)
            t_pt_gev = np.clip(t_pt_mev, 0, 5e6) * MeV

            t_eta = -np.log(np.abs(np.tan(t_theta / 2.0 + 1e-12)) + 1e-12)

            # z0 in ATLAS InDetTrackParticles is w.r.t. the beamspot, not PV.
            # For the isolation cut we apply |z0·sinθ| < 3 mm, which is
            # standard practice at ATLAS. For PV-relative z0 one would subtract
            # PVz — we keep beamspot convention here to match ATLAS reco.
            result.append({
                'pt_gev': t_pt_gev,
                'eta':    t_eta,
                'phi':    t_phi,
                'z0':     t_z0,       # mm, w.r.t. beamspot
                'theta':  t_theta,
            })

        logger.info("  Done pre-loading tracks.")
        return result

    # ── Electrons ─────────────────────────────────────────────────────────────

    def _process_electrons(self, events, track_arrays) -> Dict:
        logger.info("Processing electrons...")
        n_events = len(events)
        mx = self.config.max_objects

        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt','eta','phi','charge','trk_iso03']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        a = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['mass','LHLoose','LHMedium','LHTight',
                       'topoetcone20','ptvarcone30']}

        for i, ev in enumerate(events):
            if "AnalysisElectronsAuxDyn.pt" not in ev.fields:
                continue

            e_pt_mev = ev["AnalysisElectronsAuxDyn.pt"]
            pm = e_pt_mev > self.config.electron_pt_cut

            # Quality: require LHLoose identification
            if self.config.electron_require_loose:
                lh_loose = _safe_get(ev, "AnalysisElectronsAuxDyn.DFCommonElectronsLHLoose")
                if lh_loose is not None:
                    pm = pm & (lh_loose > 0)

            # Author cut: require author 1 (single-track) or 16 (forward)
            if self.config.electron_require_author:
                e_author = _safe_get(ev, "AnalysisElectronsAuxDyn.author")
                if e_author is not None:
                    pm = pm & ((e_author == 1) | (e_author == 16))

            # Object Quality: (OQ & 1446) == 0
            if self.config.electron_require_oq:
                e_oq = _safe_get(ev, "AnalysisElectronsAuxDyn.OQ")
                if e_oq is not None:
                    pm = pm & ((e_oq & 1446) == 0)

            # Eta cut (cluster eta, approximated by track eta here)
            e_eta = ev["AnalysisElectronsAuxDyn.eta"]
            pm = pm & (abs(e_eta) < self.config.electron_eta_cut)

            # Impact parameter: |z0*sinθ| < 0.5 mm
            if self.config.electron_z0sintheta_cut < 100:
                e_z0st = _safe_get(ev, "AnalysisElectronsAuxDyn.z0sinTheta")
                if e_z0st is not None:
                    pm = pm & (abs(e_z0st) < self.config.electron_z0sintheta_cut)

            def _s(name):
                v = _safe_get(ev, name)
                return v[pm] if v is not None else ak.zeros_like(e_pt_mev[pm])

            pt_mev  = ak.to_numpy(e_pt_mev[pm])
            eta_arr = ak.to_numpy(ev["AnalysisElectronsAuxDyn.eta"][pm])
            phi_arr = ak.to_numpy(ev["AnalysisElectronsAuxDyn.phi"][pm])

            si = np.argsort(pt_mev)[::-1]
            n  = min(len(si), mx)
            if n == 0:
                continue

            pt_gev = (pt_mev[si[:n]]) * MeV
            eta    = eta_arr[si[:n]]
            phi    = phi_arr[si[:n]]

            c['n'][i]        = n
            c['pt'][i, :n]   = pt_gev
            c['eta'][i, :n]  = eta
            c['phi'][i, :n]  = phi
            c['charge'][i, :n]  = ak.to_numpy(_s("AnalysisElectronsAuxDyn.charge"))[si[:n]]
            c['mask'][i, :n]    = True

            # Recomputed track isolation
            if track_arrays[i] is not None:
                tr = track_arrays[i]
                iso = _compute_track_iso(eta, phi, pt_gev,
                                         tr['pt_gev'], tr['eta'], tr['phi'],
                                         tr['z0'], tr['theta'])
                c['trk_iso03'][i, :n] = iso

            # ATLAS-specific
            a['mass'][i, :n]       = ak.to_numpy(_s("AnalysisElectronsAuxDyn.m"))[si[:n]] * MeV
            a['LHLoose'][i, :n]    = ak.to_numpy(_s("AnalysisElectronsAuxDyn.DFCommonElectronsLHLoose"))[si[:n]]
            a['LHMedium'][i, :n]   = ak.to_numpy(_s("AnalysisElectronsAuxDyn.DFCommonElectronsLHMedium"))[si[:n]]
            a['LHTight'][i, :n]    = ak.to_numpy(_s("AnalysisElectronsAuxDyn.DFCommonElectronsLHTight"))[si[:n]]
            a['topoetcone20'][i,:n]= ak.to_numpy(_s("AnalysisElectronsAuxDyn.topoetcone20"))[si[:n]] * MeV
            a['ptvarcone30'][i,:n] = ak.to_numpy(_s(
                "AnalysisElectronsAuxDyn.ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000"))[si[:n]] * MeV

        return {'common': c, 'atlas': a}

    # ── Muons ─────────────────────────────────────────────────────────────────

    def _process_muons(self, events, track_arrays) -> Dict:
        logger.info("Processing muons...")
        n_events = len(events)
        mx = self.config.max_objects

        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt','eta','phi','charge','trk_iso03']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        a = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['quality','muonType','passIDCuts','passPresel',
                       'topoetcone20','ptvarcone30']}

        for i, ev in enumerate(events):
            if "AnalysisMuonsAuxDyn.pt" not in ev.fields:
                continue

            mu_pt_mev = ev["AnalysisMuonsAuxDyn.pt"]
            pm = mu_pt_mev > self.config.muon_pt_cut

            # Preselection: standard ATLAS muon preselection flag
            if self.config.muon_require_presel:
                mu_presel = _safe_get(ev, "AnalysisMuonsAuxDyn.DFCommonMuonPassPreselection")
                if mu_presel is not None:
                    pm = pm & (mu_presel > 0)

            # Quality: require quality <= max (0=Tight,1=Medium,2=Loose)
            if self.config.muon_max_quality < 99:
                mu_qual = _safe_get(ev, "AnalysisMuonsAuxDyn.quality")
                if mu_qual is not None:
                    pm = pm & (mu_qual <= self.config.muon_max_quality)

            # Eta: fiducial cut
            mu_eta = ev["AnalysisMuonsAuxDyn.eta"]
            pm = pm & (abs(mu_eta) < self.config.muon_eta_cut)

            # Type-dependent cuts:
            # CaloTagged: also pT > 15 GeV
            # Combined + CaloTagged: impact parameter cuts (loose)
            # StandAlone: no impact parameter cut
            mu_type = _safe_get(ev, "AnalysisMuonsAuxDyn.muonType")
            mu_d0 = _safe_get(ev, "AnalysisMuonsAuxDyn.d0")
            mu_z0st = _safe_get(ev, "AnalysisMuonsAuxDyn.z0sinTheta")

            if mu_type is not None:
                # CaloTagged (type=3): require higher pT
                is_calo = (mu_type == 3)
                pm = pm & (~is_calo | (mu_pt_mev > self.config.muon_calo_pt_cut))

                # Combined (0) + CaloTagged (3): impact parameter cuts
                is_combined_or_calo = (mu_type == 0) | (mu_type == 3)
                if mu_d0 is not None and self.config.muon_d0_cut < 100:
                    pm = pm & (~is_combined_or_calo | (abs(mu_d0) < self.config.muon_d0_cut))
                if mu_z0st is not None and self.config.muon_z0sintheta_cut < 100:
                    pm = pm & (~is_combined_or_calo | (abs(mu_z0st) < self.config.muon_z0sintheta_cut))

            def _s(name):
                v = _safe_get(ev, name)
                return v[pm] if v is not None else ak.zeros_like(mu_pt_mev[pm])

            pt_mev  = ak.to_numpy(mu_pt_mev[pm])
            eta_arr = ak.to_numpy(ev["AnalysisMuonsAuxDyn.eta"][pm])
            phi_arr = ak.to_numpy(ev["AnalysisMuonsAuxDyn.phi"][pm])

            si = np.argsort(pt_mev)[::-1]
            n  = min(len(si), mx)
            if n == 0:
                continue

            pt_gev = pt_mev[si[:n]] * MeV
            eta    = eta_arr[si[:n]]
            phi    = phi_arr[si[:n]]

            c['n'][i]           = n
            c['pt'][i, :n]      = pt_gev
            c['eta'][i, :n]     = eta
            c['phi'][i, :n]     = phi
            c['charge'][i, :n]  = ak.to_numpy(_s("AnalysisMuonsAuxDyn.charge"))[si[:n]]
            c['mask'][i, :n]    = True

            if track_arrays[i] is not None:
                tr = track_arrays[i]
                iso = _compute_track_iso(eta, phi, pt_gev,
                                         tr['pt_gev'], tr['eta'], tr['phi'],
                                         tr['z0'], tr['theta'])
                c['trk_iso03'][i, :n] = iso

            a['quality'][i, :n]     = ak.to_numpy(_s("AnalysisMuonsAuxDyn.quality"))[si[:n]]
            a['muonType'][i, :n]    = ak.to_numpy(_s("AnalysisMuonsAuxDyn.muonType"))[si[:n]]
            a['passIDCuts'][i, :n]  = ak.to_numpy(_s("AnalysisMuonsAuxDyn.DFCommonMuonPassIDCuts"))[si[:n]]
            a['passPresel'][i, :n]  = ak.to_numpy(_s("AnalysisMuonsAuxDyn.DFCommonMuonPassPreselection"))[si[:n]]
            a['topoetcone20'][i,:n] = ak.to_numpy(_s("AnalysisMuonsAuxDyn.topoetcone20"))[si[:n]] * MeV
            a['ptvarcone30'][i, :n] = ak.to_numpy(_s("AnalysisMuonsAuxDyn.ptvarcone30"))[si[:n]] * MeV

        return {'common': c, 'atlas': a}

    # ── Photons ───────────────────────────────────────────────────────────────

    def _process_photons(self, events, track_arrays) -> Dict:
        logger.info("Processing photons...")
        n_events = len(events)
        mx = self.config.max_objects

        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt','eta','phi','trk_iso03']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        a = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['isLoose','isTight','isCleaning','author',
                       'topoetcone20','topoetcone40','ptcone20']}

        for i, ev in enumerate(events):
            if "AnalysisPhotonsAuxDyn.pt" not in ev.fields:
                continue

            ph_pt_mev = ev["AnalysisPhotonsAuxDyn.pt"]
            pm = ph_pt_mev > self.config.photon_pt_cut

            # Quality: require isLoose
            if self.config.photon_require_loose:
                ph_loose = _safe_get(ev, "AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMLoose")
                if ph_loose is not None:
                    pm = pm & (ph_loose > 0)

            def _s(name):
                v = _safe_get(ev, name)
                return v[pm] if v is not None else ak.zeros_like(ph_pt_mev[pm])

            pt_mev  = ak.to_numpy(ph_pt_mev[pm])
            eta_arr = ak.to_numpy(ev["AnalysisPhotonsAuxDyn.eta"][pm])
            phi_arr = ak.to_numpy(ev["AnalysisPhotonsAuxDyn.phi"][pm])

            si = np.argsort(pt_mev)[::-1]
            n  = min(len(si), mx)
            if n == 0:
                continue

            pt_gev = pt_mev[si[:n]] * MeV
            eta    = eta_arr[si[:n]]
            phi    = phi_arr[si[:n]]

            c['n'][i]          = n
            c['pt'][i, :n]     = pt_gev
            c['eta'][i, :n]    = eta
            c['phi'][i, :n]    = phi
            c['mask'][i, :n]   = True

            if track_arrays[i] is not None:
                tr = track_arrays[i]
                iso = _compute_track_iso(eta, phi, pt_gev,
                                         tr['pt_gev'], tr['eta'], tr['phi'],
                                         tr['z0'], tr['theta'])
                c['trk_iso03'][i, :n] = iso

            a['isLoose'][i, :n]    = ak.to_numpy(_s("AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMLoose"))[si[:n]]
            a['isTight'][i, :n]    = ak.to_numpy(_s("AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMTight"))[si[:n]]
            a['isCleaning'][i,:n]  = ak.to_numpy(_s("AnalysisPhotonsAuxDyn.DFCommonPhotonsCleaning"))[si[:n]]
            a['author'][i, :n]     = ak.to_numpy(_s("AnalysisPhotonsAuxDyn.author"))[si[:n]]
            a['topoetcone20'][i,:n]= ak.to_numpy(_s("AnalysisPhotonsAuxDyn.topoetcone20"))[si[:n]] * MeV
            a['topoetcone40'][i,:n]= ak.to_numpy(_s("AnalysisPhotonsAuxDyn.topoetcone40"))[si[:n]] * MeV
            a['ptcone20'][i, :n]   = ak.to_numpy(_s("AnalysisPhotonsAuxDyn.ptcone20"))[si[:n]] * MeV

        return {'common': c, 'atlas': a}

    # ── Taus ──────────────────────────────────────────────────────────────────

    def _process_taus(self, events) -> Dict:
        logger.info("Processing taus...")
        n_events = len(events)
        mx = self.config.max_objects

        # common: kinematics + charge + topology flag portable to CMS
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt','eta','phi','charge','is_1prong']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        # atlas-specific: discriminant scores
        a = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in [
                 'NNDecayMode',
                 'RNNJetScore',
                 'RNNEleScore', 'EleRNNLoose_v1',
                 'EleRNNMedium_v1', 'EleRNNTight_v1',
             ]}

        for i, ev in enumerate(events):
            if "AnalysisTauJetsAuxDyn.pt" not in ev.fields:
                continue

            tau_pt_mev = ev["AnalysisTauJetsAuxDyn.pt"]
            pm = tau_pt_mev > self.config.tau_pt_cut

            # Quality: require RNNJetScore above loose WP (removes QCD fakes)
            if self.config.tau_rnn_jet_cut > 0:
                rnn_jet = _safe_get(ev, "AnalysisTauJetsAuxDyn.RNNJetScore")
                if rnn_jet is not None:
                    pm = pm & (rnn_jet > self.config.tau_rnn_jet_cut)

            def _s(name):
                v = _safe_get(ev, name)
                return v[pm] if v is not None else ak.zeros_like(tau_pt_mev[pm])

            pt_mev  = ak.to_numpy(tau_pt_mev[pm])
            eta_arr = ak.to_numpy(ev["AnalysisTauJetsAuxDyn.eta"][pm])
            phi_arr = ak.to_numpy(ev["AnalysisTauJetsAuxDyn.phi"][pm])

            nn_dm      = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.NNDecayMode"))
            decay_mode = nn_dm

            si = np.argsort(pt_mev)[::-1]
            n  = min(len(si), mx)
            if n == 0:
                continue

            c['n'][i]          = n
            c['pt'][i, :n]     = pt_mev[si[:n]] * MeV
            c['eta'][i, :n]    = eta_arr[si[:n]]
            c['phi'][i, :n]    = phi_arr[si[:n]]
            c['charge'][i, :n] = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.charge"))[si[:n]]
            c['mask'][i, :n]   = True

            # is_1prong: 1-prong modes are 0,1,2; 3-prong are 10,11 — same as CMS Tau_decayMode
            dm_sel = decay_mode[si[:n]]
            c['is_1prong'][i, :n] = np.where(dm_sel <= 2, 1.0,
                                    np.where(dm_sel >= 10, 0.0, -1.0))

            # ATLAS-specific
            a['NNDecayMode'][i, :n]     = nn_dm[si[:n]]
            a['RNNJetScore'][i, :n]     = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.RNNJetScore"))[si[:n]]
            a['RNNEleScore'][i, :n]     = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.RNNEleScore"))[si[:n]]
            a['EleRNNLoose_v1'][i, :n]  = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.EleRNNLoose_v1"))[si[:n]]
            a['EleRNNMedium_v1'][i, :n] = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.EleRNNMedium_v1"))[si[:n]]
            a['EleRNNTight_v1'][i, :n]  = ak.to_numpy(_s("AnalysisTauJetsAuxDyn.EleRNNTight_v1"))[si[:n]]

        return {'common': c, 'atlas': a}

    # ── Jets ──────────────────────────────────────────────────────────────────

    def _process_jets(self, events) -> Dict:
        logger.info("Processing jets...")
        n_events = len(events)
        mx = self.config.max_objects

        # common: 4-vector + n_trk (closest to CMS Jet_nConstituents)
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt','eta','phi','mass','n_trk']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        # atlas-specific: b-tagging scores + QG tagger
        a = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['DL1d_pb','DL1d_pc','DL1d_pu',
                       'GN2_pb','GN2_pc','GN2_pu',
                       'QG_nTracks','QG_tracksWidth','QG_tracksC1']}

        for i, ev in enumerate(events):
            if "AnalysisJetsAuxDyn.pt" not in ev.fields:
                continue

            j_pt_mev = ev["AnalysisJetsAuxDyn.pt"]
            j_eta = ev["AnalysisJetsAuxDyn.eta"]
            pm = (j_pt_mev > self.config.jet_pt_cut) & (abs(j_eta) < self.config.jet_eta_cut)

            # Jet cleaning: LooseBad flag
            if self.config.jet_clean_loose:
                j_clean = _safe_get(ev, "AnalysisJetsAuxDyn.DFCommonJets_jetClean_LooseBad")
                if j_clean is not None:
                    pm = pm & (j_clean > 0)

            def _s(name):
                v = _safe_get(ev, name)
                if v is None:
                    return ak.zeros_like(j_pt_mev[pm])
                try:
                    # NumTrkPt500 is nested: (n_jets, n_vertices) — take first vertex
                    if ak.min(ak.num(v, axis=1)) is not None:
                        v = v[:, 0]
                except Exception:
                    pass
                return v[pm] if len(v) == len(j_pt_mev) else ak.zeros_like(j_pt_mev[pm])

            pt_mev  = ak.to_numpy(j_pt_mev[pm])
            eta_arr = ak.to_numpy(ev["AnalysisJetsAuxDyn.eta"][pm])
            phi_arr = ak.to_numpy(ev["AnalysisJetsAuxDyn.phi"][pm])
            m_arr   = ak.to_numpy(ev["AnalysisJetsAuxDyn.m"][pm])

            si = np.argsort(pt_mev)[::-1]
            n  = min(len(si), mx)
            if n == 0:
                continue

            c['n'][i]          = n
            c['pt'][i, :n]     = pt_mev[si[:n]] * MeV
            c['eta'][i, :n]    = eta_arr[si[:n]]
            c['phi'][i, :n]    = phi_arr[si[:n]]
            c['mass'][i, :n]   = m_arr[si[:n]] * MeV
            c['n_trk'][i, :n]  = ak.to_numpy(_s("AnalysisJetsAuxDyn.NumTrkPt500"))[si[:n]]
            c['mask'][i, :n]   = True

            a['DL1d_pb'][i, :n] = ak.to_numpy(_s("BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb"))[si[:n]]
            a['DL1d_pc'][i, :n] = ak.to_numpy(_s("BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc"))[si[:n]]
            a['DL1d_pu'][i, :n] = ak.to_numpy(_s("BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu"))[si[:n]]
            a['GN2_pb'][i, :n]  = ak.to_numpy(_s("BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pb"))[si[:n]]
            a['GN2_pc'][i, :n]  = ak.to_numpy(_s("BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pc"))[si[:n]]
            a['GN2_pu'][i, :n]  = ak.to_numpy(_s("BTagging_AntiKt4EMPFlowAuxDyn.GN2v00_pu"))[si[:n]]
            a['QG_nTracks'][i, :n]     = ak.to_numpy(_s("AnalysisJetsAuxDyn.DFCommonJets_QGTagger_NTracks"))[si[:n]]
            a['QG_tracksWidth'][i, :n] = ak.to_numpy(_s("AnalysisJetsAuxDyn.DFCommonJets_QGTagger_TracksWidth"))[si[:n]]
            a['QG_tracksC1'][i, :n]    = ak.to_numpy(_s("AnalysisJetsAuxDyn.DFCommonJets_QGTagger_TracksC1"))[si[:n]]

        return {'common': c, 'atlas': a}

    # ── Tracks ────────────────────────────────────────────────────────────────

    def _process_tracks(self, events) -> Dict:
        logger.info("Processing tracks...")
        n_events = len(events)
        mx = self.config.max_tracks

        # common: minimal 5-parameter track
        c = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['pt','eta','phi','d0','z0']}
        c['n'] = np.zeros(n_events, dtype=np.int32)
        c['mask'] = np.zeros((n_events, mx), dtype=bool)

        # atlas-specific: qOverP and track quality
        a = {k: np.zeros((n_events, mx), dtype=np.float32)
             for k in ['qOverP','chiSquared','nDoF']}

        for i, ev in enumerate(events):
            if "InDetTrackParticlesAuxDyn.qOverP" not in ev.fields:
                continue

            t_qop   = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.qOverP"])
            t_theta = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.theta"])
            t_phi   = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.phi"])
            t_d0    = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.d0"]) \
                      if "InDetTrackParticlesAuxDyn.d0" in ev.fields else np.zeros_like(t_qop)
            t_z0    = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.z0"]) \
                      if "InDetTrackParticlesAuxDyn.z0" in ev.fields else np.zeros_like(t_qop)

            abs_qop = np.abs(t_qop) + 1e-12
            sin_th  = np.sin(t_theta)
            t_pt_mev = np.abs(sin_th / abs_qop)
            t_pt_mev = np.clip(t_pt_mev, 0, 5e6)
            t_eta_all = -np.log(np.abs(np.tan(t_theta / 2.0 + 1e-12)) + 1e-12)

            # Track quality: pT + eta + impact parameter cuts
            pm = t_pt_mev > self.config.track_pt_cut
            pm = pm & (np.abs(t_eta_all) < self.config.track_eta_cut)
            if self.config.track_d0_cut < 100:
                pm = pm & (np.abs(t_d0) < self.config.track_d0_cut)
            if self.config.track_z0sintheta_cut < 100:
                z0sinth = np.abs(t_z0 * sin_th)
                pm = pm & (z0sinth < self.config.track_z0sintheta_cut)

            t_pt_gev = t_pt_mev[pm] * MeV
            t_eta    = t_eta_all[pm]

            si = np.argsort(t_pt_gev)[::-1]
            n  = min(len(si), mx)
            if n == 0:
                continue

            c['n'][i]         = n
            c['pt'][i, :n]    = t_pt_gev[si[:n]]
            c['eta'][i, :n]   = t_eta[si[:n]]
            c['phi'][i, :n]   = t_phi[pm][si[:n]]
            c['d0'][i, :n]    = t_d0[pm][si[:n]]
            c['z0'][i, :n]    = t_z0[pm][si[:n]]
            c['mask'][i, :n]  = True

            a['qOverP'][i, :n] = t_qop[pm][si[:n]]
            if "InDetTrackParticlesAuxDyn.chiSquared" in ev.fields:
                a['chiSquared'][i, :n] = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.chiSquared"])[pm][si[:n]]
            if "InDetTrackParticlesAuxDyn.numberDoF" in ev.fields:
                a['nDoF'][i, :n] = ak.to_numpy(ev["InDetTrackParticlesAuxDyn.numberDoF"])[pm][si[:n]]

        return {'common': c, 'atlas': a}

    # ── MET ───────────────────────────────────────────────────────────────────

    def _process_met(self, events) -> Dict:
        logger.info("Processing MET...")
        n_events = len(events)
        c = {k: np.zeros(n_events, dtype=np.float32) for k in ['pt','phi','sumet']}

        def _first(key):
            if key not in events.fields:
                return np.zeros(n_events, dtype=np.float32)
            arr = events[key]
            try:
                return ak.to_numpy(arr[:, 0]).astype(np.float32)
            except Exception:
                out = np.zeros(n_events, dtype=np.float32)
                for j in range(n_events):
                    try:
                        v = arr[j]
                        if len(v) > 0:
                            out[j] = float(v[0])
                    except Exception:
                        pass
                return out

        mpx   = _first("MET_Core_AnalysisMETAuxDyn.mpx")
        mpy   = _first("MET_Core_AnalysisMETAuxDyn.mpy")
        sumet = _first("MET_Core_AnalysisMETAuxDyn.sumet")

        c['pt']    = np.sqrt(mpx**2 + mpy**2) * MeV
        c['phi']   = np.arctan2(mpy, mpx).astype(np.float32)
        c['sumet'] = sumet * MeV

        logger.info(f"  MET mean={c['pt'].mean():.2f} GeV, nonzero={np.count_nonzero(c['pt'])}/{n_events}")
        return c   # MET has no ATLAS-specific split — everything goes to common

    # ── Event info ────────────────────────────────────────────────────────────

    def _process_event_info(self, events) -> Dict:
        logger.info("Processing event info...")
        n_events = len(events)

        # common
        c = {k: np.zeros(n_events, dtype=np.float32)
             for k in ['pvx','pvy','pvz','mu']}
        c['experiment_id'] = np.full(n_events, EXPERIMENT_ID, dtype=np.int8)
        c['is_simulation'] = np.zeros(n_events, dtype=np.int8)

        # atlas-specific
        a = {
            'event_number':    np.zeros(n_events, dtype=np.int64),
            'run_number':      np.zeros(n_events, dtype=np.int32),
            'mcChannelNumber': np.zeros(n_events, dtype=np.int32),
        }

        for i, ev in enumerate(events):
            if "EventInfoAuxDyn.averageInteractionsPerCrossing" in ev.fields:
                c['mu'][i] = float(ev["EventInfoAuxDyn.averageInteractionsPerCrossing"])
            if "EventInfoAuxDyn.eventNumber" in ev.fields:
                a['event_number'][i] = int(ev["EventInfoAuxDyn.eventNumber"])
            if "EventInfoAuxDyn.runNumber" in ev.fields:
                a['run_number'][i] = int(ev["EventInfoAuxDyn.runNumber"])
            mc = 0
            if "EventInfoAuxDyn.mcChannelNumber" in ev.fields:
                mc = int(ev["EventInfoAuxDyn.mcChannelNumber"])
            a['mcChannelNumber'][i] = mc
            c['is_simulation'][i]   = 1 if mc > 0 else 0

            # Primary vertex (vertexType == 1)
            if "PrimaryVerticesAuxDyn.x" in ev.fields:
                vx = ak.to_numpy(ev["PrimaryVerticesAuxDyn.x"])
                vy = ak.to_numpy(ev["PrimaryVerticesAuxDyn.y"])
                vz = ak.to_numpy(ev["PrimaryVerticesAuxDyn.z"])
                vt = _safe_get(ev, "PrimaryVerticesAuxDyn.vertexType")
                idx = 0
                if vt is not None:
                    where = np.where(ak.to_numpy(vt) == 1)[0]
                    if len(where) > 0:
                        idx = where[0]
                if len(vx) > 0:
                    c['pvx'][i] = float(vx[idx])
                    c['pvy'][i] = float(vy[idx])
                    c['pvz'][i] = float(vz[idx])

        return {'common': c, 'atlas': a}

    def _process_truth(self, events) -> Dict:
        """Process truth particles (MC only). Stores truth leptons, photons, jets, bosons, MET."""
        logger.info("Processing truth particles...")
        n_events = len(events)
        mx = self.config.max_objects

        truth_data = {}

        # Truth particles: electrons, muons, photons, taus
        truth_particles = {
            'electrons': ('TruthElectronsAuxDyn', mx),
            'muons':     ('TruthMuonsAuxDyn', mx),
            'photons':   ('TruthPhotonsAuxDyn', mx),
            'taus':      ('TruthTausAuxDyn', mx),
        }

        for obj_name, (prefix, max_n) in truth_particles.items():
            pt_key = f"{prefix}.pt"
            if pt_key not in events.fields:
                continue

            d = {
                'pt':     np.zeros((n_events, max_n), dtype=np.float32),
                'eta':    np.zeros((n_events, max_n), dtype=np.float32),
                'phi':    np.zeros((n_events, max_n), dtype=np.float32),
                'mass':   np.zeros((n_events, max_n), dtype=np.float32),
                'pdgId':  np.zeros((n_events, max_n), dtype=np.int32),
                'status': np.zeros((n_events, max_n), dtype=np.int32),
                'n':      np.zeros(n_events, dtype=np.int32),
                'mask':   np.zeros((n_events, max_n), dtype=bool),
            }

            for i, ev in enumerate(events):
                pt_arr = _safe_get(ev, pt_key)
                if pt_arr is None:
                    continue
                pt_np = ak.to_numpy(pt_arr)
                n = min(len(pt_np), max_n)
                if n == 0:
                    continue
                si = np.argsort(pt_np)[::-1][:n]
                d['pt'][i, :n]     = pt_np[si] * MeV
                d['eta'][i, :n]    = ak.to_numpy(_safe_get(ev, f"{prefix}.eta"))[si]
                d['phi'][i, :n]    = ak.to_numpy(_safe_get(ev, f"{prefix}.phi"))[si]
                m = _safe_get(ev, f"{prefix}.m")
                if m is not None:
                    d['mass'][i, :n] = ak.to_numpy(m)[si] * MeV
                pid = _safe_get(ev, f"{prefix}.pdgId")
                if pid is not None:
                    d['pdgId'][i, :n] = ak.to_numpy(pid)[si]
                st = _safe_get(ev, f"{prefix}.status")
                if st is not None:
                    d['status'][i, :n] = ak.to_numpy(st)[si]
                d['n'][i] = n
                d['mask'][i, :n] = True

            truth_data[obj_name] = d
            logger.info(f"  Truth {obj_name}: avg {d['n'].mean():.1f} per event")

        # Truth bosons (Higgs, Z, W)
        boson_prefix = "TruthBosonAuxDyn"
        max_bosons = 5
        if f"{boson_prefix}.pt" in events.fields:
            d = {
                'pt':     np.zeros((n_events, max_bosons), dtype=np.float32),
                'eta':    np.zeros((n_events, max_bosons), dtype=np.float32),
                'phi':    np.zeros((n_events, max_bosons), dtype=np.float32),
                'mass':   np.zeros((n_events, max_bosons), dtype=np.float32),
                'pdgId':  np.zeros((n_events, max_bosons), dtype=np.int32),
                'status': np.zeros((n_events, max_bosons), dtype=np.int32),
                'n':      np.zeros(n_events, dtype=np.int32),
                'mask':   np.zeros((n_events, max_bosons), dtype=bool),
            }
            for i, ev in enumerate(events):
                pt_arr = _safe_get(ev, f"{boson_prefix}.pt")
                if pt_arr is None:
                    continue
                pt_np = ak.to_numpy(pt_arr)
                n = min(len(pt_np), max_bosons)
                if n == 0:
                    continue
                si = np.argsort(pt_np)[::-1][:n]
                d['pt'][i, :n]   = pt_np[si] * MeV
                d['eta'][i, :n]  = ak.to_numpy(_safe_get(ev, f"{boson_prefix}.eta"))[si]
                d['phi'][i, :n]  = ak.to_numpy(_safe_get(ev, f"{boson_prefix}.phi"))[si]
                bm = _safe_get(ev, f"{boson_prefix}.m")
                if bm is not None:
                    d['mass'][i, :n] = ak.to_numpy(bm)[si] * MeV
                pid = _safe_get(ev, f"{boson_prefix}.pdgId")
                if pid is not None:
                    d['pdgId'][i, :n] = ak.to_numpy(pid)[si]
                st = _safe_get(ev, f"{boson_prefix}.status")
                if st is not None:
                    d['status'][i, :n] = ak.to_numpy(st)[si]
                d['n'][i] = n
                d['mask'][i, :n] = True
            truth_data['bosons'] = d
            logger.info(f"  Truth bosons: avg {d['n'].mean():.1f} per event")

        # Truth jets
        jet_prefix = "AntiKt4TruthDressedWZJetsAuxDyn"
        if f"{jet_prefix}.pt" in events.fields:
            d = {
                'pt':   np.zeros((n_events, mx), dtype=np.float32),
                'eta':  np.zeros((n_events, mx), dtype=np.float32),
                'phi':  np.zeros((n_events, mx), dtype=np.float32),
                'mass': np.zeros((n_events, mx), dtype=np.float32),
                'n':    np.zeros(n_events, dtype=np.int32),
                'mask': np.zeros((n_events, mx), dtype=bool),
            }
            for i, ev in enumerate(events):
                pt_arr = _safe_get(ev, f"{jet_prefix}.pt")
                if pt_arr is None:
                    continue
                pt_np = ak.to_numpy(pt_arr)
                pm = pt_np > 20_000  # 20 GeV truth jet cut
                pt_np = pt_np[pm]
                n = min(len(pt_np), mx)
                if n == 0:
                    continue
                si = np.argsort(pt_np)[::-1][:n]
                d['pt'][i, :n]   = pt_np[si] * MeV
                eta_all = ak.to_numpy(_safe_get(ev, f"{jet_prefix}.eta"))[pm]
                phi_all = ak.to_numpy(_safe_get(ev, f"{jet_prefix}.phi"))[pm]
                d['eta'][i, :n]  = eta_all[si]
                d['phi'][i, :n]  = phi_all[si]
                jm = _safe_get(ev, f"{jet_prefix}.m")
                if jm is not None:
                    d['mass'][i, :n] = ak.to_numpy(jm)[pm][si] * MeV
                d['n'][i] = n
                d['mask'][i, :n] = True
            truth_data['jets'] = d
            logger.info(f"  Truth jets: avg {d['n'].mean():.1f} per event")

        # Truth MET
        if "MET_TruthAuxDyn.mpx" in events.fields:
            d = {
                'pt':    np.zeros(n_events, dtype=np.float32),
                'phi':   np.zeros(n_events, dtype=np.float32),
                'sumet': np.zeros(n_events, dtype=np.float32),
            }
            for i, ev in enumerate(events):
                mpx = _safe_get(ev, "MET_TruthAuxDyn.mpx")
                mpy = _safe_get(ev, "MET_TruthAuxDyn.mpy")
                sumet = _safe_get(ev, "MET_TruthAuxDyn.sumet")
                if mpx is not None and mpy is not None:
                    mpx_np = ak.to_numpy(mpx)
                    mpy_np = ak.to_numpy(mpy)
                    if mpx_np.ndim > 0 and len(mpx_np) > 0:
                        mx_val = float(mpx_np[-1]) * MeV
                        my_val = float(mpy_np[-1]) * MeV
                        d['pt'][i] = np.sqrt(mx_val**2 + my_val**2)
                        d['phi'][i] = np.arctan2(my_val, mx_val)
                if sumet is not None:
                    sumet_np = ak.to_numpy(sumet)
                    if sumet_np.ndim > 0 and len(sumet_np) > 0:
                        d['sumet'][i] = float(sumet_np[-1]) * MeV
            truth_data['met'] = d
            logger.info(f"  Truth MET: avg {d['pt'].mean():.1f} GeV")

        return truth_data

    # ── HDF5 writer ───────────────────────────────────────────────────────────

    def save_to_hdf5(self, data: Dict, output_file: str, input_file: str):
        logger.info(f"Writing: {output_file}")
        cfg = self.config
        n_events = len(data['event_info']['common']['experiment_id'])

        with h5py.File(output_file, 'w') as f:

            # ── /common ──────────────────────────────────────────────────────
            grp_c = f.create_group('common')

            for obj in ['electrons','muons','photons','taus','jets','tracks']:
                sub = grp_c.create_group(obj)
                for key, arr in data[obj]['common'].items():
                    sub.create_dataset(key, data=arr, compression='gzip',
                                       compression_opts=4)

            # MET is entirely common
            grp_met = grp_c.create_group('met')
            for key, arr in data['met'].items():
                grp_met.create_dataset(key, data=arr, compression='gzip')

            # Event common
            grp_ev = grp_c.create_group('event')
            for key, arr in data['event_info']['common'].items():
                grp_ev.create_dataset(key, data=arr, compression='gzip')

            # ── /atlas ───────────────────────────────────────────────────────
            grp_a = f.create_group('atlas')

            for obj in ['electrons','muons','photons','taus','jets','tracks']:
                sub = grp_a.create_group(obj)
                for key, arr in data[obj]['atlas'].items():
                    sub.create_dataset(key, data=arr, compression='gzip',
                                       compression_opts=4)

            grp_aev = grp_a.create_group('event')
            for key, arr in data['event_info']['atlas'].items():
                grp_aev.create_dataset(key, data=arr, compression='gzip')

            # ── /truth (MC only) ──────────────────────────────────────────
            if 'truth' in data and data['truth']:
                grp_t = f.create_group('truth')
                for obj_name, obj_dict in data['truth'].items():
                    if obj_name == 'met':
                        # MET is 1D per event, not per-object
                        sub = grp_t.create_group('met')
                        for key, arr in obj_dict.items():
                            sub.create_dataset(key, data=arr, compression='gzip')
                    else:
                        sub = grp_t.create_group(obj_name)
                        for key, arr in obj_dict.items():
                            sub.create_dataset(key, data=arr, compression='gzip',
                                               compression_opts=4)
                logger.info(f"  Truth groups: {list(data['truth'].keys())}")

            # ── /metadata  (attrs only — no datasets) ────────────────────────
            meta = f.create_group('metadata')
            meta.attrs['experiment']            = 'ATLAS'
            meta.attrs['experiment_id']         = EXPERIMENT_ID
            meta.attrs['converter_version']     = CONVERTER_VERSION
            meta.attrs['converter_changelog']   = CONVERTER_CHANGELOG.get(CONVERTER_VERSION, '')
            meta.attrs['input_file']            = str(Path(input_file).name)
            meta.attrs['n_events']              = n_events

            # ── CERN Open Data metadata ──────────────────────────────
            fname = Path(input_file).name
            dsid = _extract_dsid(fname)

            # If DSID not in filename, get from mcChannelNumber in data
            if dsid is None:
                mc_ch = data['event_info']['atlas'].get('mcChannelNumber')
                if mc_ch is not None and len(mc_ch) > 0:
                    unique_mc = np.unique(mc_ch[mc_ch > 0])
                    if len(unique_mc) == 1:
                        dsid = int(unique_mc[0])

            mc_info = _lookup_mc_metadata(dsid) if dsid else {}

            meta.attrs['dsid']              = dsid or 0
            meta.attrs['is_mc']             = bool(mc_info.get('is_mc', dsid is not None and dsid > 0))
            meta.attrs['process']           = mc_info.get('process', '')
            meta.attrs['generator']         = mc_info.get('generator', '')
            meta.attrs['cross_section_pb']  = mc_info.get('cross_section_pb', 0.0)
            meta.attrs['filter_efficiency'] = mc_info.get('filter_efficiency', 1.0)
            meta.attrs['k_factor']          = mc_info.get('k_factor', 1.0)
            meta.attrs['keywords']          = mc_info.get('keywords', '')
            meta.attrs['sqrt_s_tev']        = 13.0
            meta.attrs['data_taking_year']  = '2015-2016'
            meta.attrs['mc_campaign']       = 'mc20a'
            meta.attrs['athena_release']    = 'Release 22'
            meta.attrs['data_format']       = 'DAOD_PHYSLITE'
            meta.attrs['license']           = 'CC0-1.0'
            meta.attrs['doi']               = 'https://doi.org/10.7483/OPENDATA.ATLAS.9HK7.P5SI'
            meta.attrs['citation']          = ('ATLAS collaboration (2024). DAOD_PHYSLITE format '
                                               '2015-2016 Open Data for Research from the ATLAS '
                                               'experiment. CERN Open Data Portal.')

            if dsid:
                logger.info(f"  DSID: {dsid}, process: {mc_info.get('process','unknown')}, "
                            f"xsec={mc_info.get('cross_section_pb',0):.4f} pb")

            # Object selection cuts
            meta.attrs['electron_pt_cut_gev']   = cfg.electron_pt_cut * MeV
            meta.attrs['muon_pt_cut_gev']       = cfg.muon_pt_cut     * MeV
            meta.attrs['photon_pt_cut_gev']     = cfg.photon_pt_cut   * MeV
            meta.attrs['tau_pt_cut_gev']        = cfg.tau_pt_cut      * MeV
            meta.attrs['jet_pt_cut_gev']        = cfg.jet_pt_cut      * MeV
            meta.attrs['track_pt_cut_gev']      = cfg.track_pt_cut    * MeV
            meta.attrs['max_objects']           = cfg.max_objects
            meta.attrs['max_tracks']            = cfg.max_tracks

            # Quality selections (HZZ-style)
            meta.attrs['electron_require_loose'] = cfg.electron_require_loose
            meta.attrs['electron_eta_cut']       = cfg.electron_eta_cut
            meta.attrs['electron_z0sintheta_cut_mm'] = cfg.electron_z0sintheta_cut
            meta.attrs['electron_require_author'] = cfg.electron_require_author
            meta.attrs['electron_require_oq']    = cfg.electron_require_oq
            meta.attrs['muon_max_quality']       = cfg.muon_max_quality
            meta.attrs['muon_require_presel']    = cfg.muon_require_presel
            meta.attrs['muon_eta_cut']           = cfg.muon_eta_cut
            meta.attrs['muon_d0_cut_mm']         = cfg.muon_d0_cut
            meta.attrs['muon_z0sintheta_cut_mm'] = cfg.muon_z0sintheta_cut
            meta.attrs['muon_calo_pt_cut_gev']   = cfg.muon_calo_pt_cut * MeV
            meta.attrs['photon_require_loose']   = cfg.photon_require_loose
            meta.attrs['tau_rnn_jet_cut']        = cfg.tau_rnn_jet_cut
            meta.attrs['jet_eta_cut']            = cfg.jet_eta_cut
            meta.attrs['jet_clean_loose']        = cfg.jet_clean_loose
            meta.attrs['track_eta_cut']          = cfg.track_eta_cut
            meta.attrs['track_d0_cut_mm']        = cfg.track_d0_cut
            meta.attrs['track_z0sintheta_cut_mm'] = cfg.track_z0sintheta_cut

            # Isolation definition — must match CMS converter
            meta.attrs['common_iso_cone']             = ISO_CONE
            meta.attrs['common_iso_pt_floor_gev']     = ISO_PT_FLOOR_GEV
            meta.attrs['common_iso_z0sintheta_cut_mm'] = ISO_Z0ST_CUT_MM
            meta.attrs['common_iso_note'] = (
                "Track isolation computed from InDetTrackParticles. "
                "z0 is w.r.t. beamspot (ATLAS convention); |z0*sinθ| < 3mm applied. "
                "CMS uses PackedCandidates PV-associated tracks — distributions differ."
            )

            # Unit declarations
            meta.attrs['units_pt']     = 'GeV'
            meta.attrs['units_energy'] = 'GeV'
            meta.attrs['units_angles'] = 'radians'
            meta.attrs['units_d0_z0']  = 'mm'
            meta.attrs['units_iso']    = 'dimensionless (sum_pT / object_pT)'

            # Schema documentation as JSON string
            schema = {
                'common': {
                    'electrons': ['pt','eta','phi','charge','trk_iso03','mask','n'],
                    'muons':     ['pt','eta','phi','charge','trk_iso03','mask','n'],
                    'taus':      ['pt','eta','phi','charge','is_1prong','mask','n'],
                    'photons':   ['pt','eta','phi','trk_iso03','mask','n'],
                    'jets':      ['pt','eta','phi','mass','n_trk','mask','n'],
                    'tracks':    ['pt','eta','phi','d0','z0','mask','n'],
                    'met':       ['pt','phi','sumet'],
                    'event':     ['pvx','pvy','pvz','mu','experiment_id','is_simulation'],
                },
                'atlas': {
                    'electrons': ['mass','LHLoose','LHMedium','LHTight','topoetcone20','ptvarcone30'],
                    'muons':     ['quality','muonType','passIDCuts','passPresel','topoetcone20','ptvarcone30'],
                    'taus':      ['NNDecayMode','RNNJetScore','RNNEleScore','EleRNNLoose_v1','EleRNNMedium_v1','EleRNNTight_v1'],
                    'photons':   ['isLoose','isTight','isCleaning','author','topoetcone20','topoetcone40','ptcone20'],
                    'jets':      ['DL1d_pb','DL1d_pc','DL1d_pu','GN2_pb','GN2_pc','GN2_pu','QG_nTracks','QG_tracksWidth','QG_tracksC1'],
                    'tracks':    ['qOverP','chiSquared','nDoF'],
                    'event':     ['event_number','run_number','mcChannelNumber'],
                },
            }
            meta.attrs['schema_json'] = json.dumps(schema)

        logger.info(f"Saved {n_events} events → {output_file}")
        self._print_summary(output_file)

    def _print_summary(self, path: str):
        with h5py.File(path, 'r') as f:
            logger.info("── File summary ─────────────────────────────────")
            for top in ['common', 'atlas']:
                for obj in f[top]:
                    grp = f[f'{top}/{obj}']
                    keys = list(grp.keys())
                    shapes = {k: grp[k].shape for k in keys[:3]}
                    logger.info(f"  /{top}/{obj}: {keys}")
            logger.info(f"  /metadata attrs: {dict(f['metadata'].attrs)}")

    # ── Entry point ───────────────────────────────────────────────────────────

    def convert(self, input_file: str, output_file: str):
        data = self.read_root_file(input_file)
        self.save_to_hdf5(data, output_file, input_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert ATLAS DAOD_PHYSLITE → HDF5 (cross-experiment schema)')
    parser.add_argument('input',  help='Input ROOT file')
    parser.add_argument('output', help='Output HDF5 file')
    parser.add_argument('--electron-pt-cut', type=float, default=7_000,  help='MeV')
    parser.add_argument('--muon-pt-cut',     type=float, default=5_000,  help='MeV')
    parser.add_argument('--photon-pt-cut',   type=float, default=10_000, help='MeV')
    parser.add_argument('--tau-pt-cut',      type=float, default=20_000, help='MeV')
    parser.add_argument('--jet-pt-cut',      type=float, default=20_000, help='MeV')
    parser.add_argument('--track-pt-cut',    type=float, default=500,    help='MeV')
    parser.add_argument('--max-objects',     type=int,   default=20)
    parser.add_argument('--max-tracks',      type=int,   default=50)
    args = parser.parse_args()

    cfg = PhysicsObjectConfig(
        electron_pt_cut=args.electron_pt_cut,
        muon_pt_cut=args.muon_pt_cut,
        photon_pt_cut=args.photon_pt_cut,
        tau_pt_cut=args.tau_pt_cut,
        jet_pt_cut=args.jet_pt_cut,
        track_pt_cut=args.track_pt_cut,
        max_objects=args.max_objects,
        max_tracks=args.max_tracks,
    )
    DAOD_PHYSLITE_Converter(cfg).convert(args.input, args.output)


if __name__ == "__main__":
    main()
  
