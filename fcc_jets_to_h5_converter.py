import argparse
import json
import h5py
import numpy as np
import uproot
import awkward as ak

PFCAND_FEATURES = [
    "pfcand_erel_log", "pfcand_thetarel", "pfcand_phirel",
    "pfcand_dptdpt", "pfcand_detadeta", "pfcand_dphidphi",
    "pfcand_dxydxy", "pfcand_dzdz", "pfcand_dxydz",
    "pfcand_dphidxy", "pfcand_dlambdadz", "pfcand_dxyc",
    "pfcand_dxyctgtheta", "pfcand_phic", "pfcand_phidz",
    "pfcand_phictgtheta", "pfcand_cdz", "pfcand_cctgtheta",
    "pfcand_mtof", "pfcand_dndx", "pfcand_charge",
    "pfcand_isMu", "pfcand_isEl", "pfcand_isChargedHad",
    "pfcand_isGamma", "pfcand_isNeutralHad", "pfcand_type",
    "pfcand_dxy", "pfcand_dz",
    "pfcand_btagSip2dVal", "pfcand_btagSip2dSig",
    "pfcand_btagSip3dVal", "pfcand_btagSip3dSig",
    "pfcand_btagJetDistVal", "pfcand_btagJetDistSig",
    "pfcand_e", "pfcand_p", "pfcand_theta", "pfcand_phi",
]

LABEL_BRANCHES = [
    "recojet_isG", "recojet_isU", "recojet_isD", "recojet_isS",
    "recojet_isC", "recojet_isB", "recojet_isTAU",
]


def theta_to_eta(theta):
    theta_clipped = np.clip(theta, 1e-7, np.pi - 1e-7)
    return -np.log(np.tan(theta_clipped / 2.0))


class FCCConverter:
    def __init__(self, input_file, tree_name="tree", max_objects=20, max_tracks=50, experiment_id=2):
        self.input_file = input_file
        self.tree_name = tree_name
        self.max_objects = max_objects
        self.max_tracks = max_tracks
        self.experiment_id = experiment_id

    def convert(self, output_file, pad_missing_common=False):
        with uproot.open(f"{self.input_file}:{self.tree_name}") as tree:
            jet_branches = [
                "jet_p", "jet_e", "jet_mass", "jet_phi", "jet_theta",
                "jet_nconst", "jet_nmu", "jet_nel", "jet_nchad",
                "jet_ngamma", "jet_nnhad", "jet_npfcand",
            ]
            arrays = tree.arrays(LABEL_BRANCHES + jet_branches + PFCAND_FEATURES, library="ak")

        # Pair jets by 2 (exclusive 2-jet clustering -> 2 entries per event)
        n_events = len(arrays) // 2
        arrays = arrays[: 2 * n_events]

        # --- 1. common/event ---
        common_event = {
            "experiment_id": np.full(n_events, self.experiment_id, dtype=np.int8),
            "is_simulation": np.ones(n_events, dtype=np.int8),
            "mu": np.zeros(n_events, dtype=np.float32),
            "pvx": np.zeros(n_events, dtype=np.float32),
            "pvy": np.zeros(n_events, dtype=np.float32),
            "pvz": np.zeros(n_events, dtype=np.float32),
        }

        # --- 2. common/jets (2 jets per event in slots 0 and 1, keeping full p) ---
        jet_p_val = ak.to_numpy(arrays["jet_p"]).astype(np.float32).reshape(n_events, 2)
        jet_theta_val = ak.to_numpy(arrays["jet_theta"]).astype(np.float32).reshape(n_events, 2)
        jet_eta_val = theta_to_eta(jet_theta_val)
        jet_phi_val = ak.to_numpy(arrays["jet_phi"]).astype(np.float32).reshape(n_events, 2)
        jet_mass_val = ak.to_numpy(arrays["jet_mass"]).astype(np.float32).reshape(n_events, 2)
        jet_n_trk_val = (
            ak.to_numpy(arrays["jet_nchad"])
            + ak.to_numpy(arrays["jet_nel"])
            + ak.to_numpy(arrays["jet_nmu"])
        ).astype(np.float32).reshape(n_events, 2)

        common_jets = {
            "p": np.zeros((n_events, self.max_objects), dtype=np.float32),
            "eta": np.zeros((n_events, self.max_objects), dtype=np.float32),
            "phi": np.zeros((n_events, self.max_objects), dtype=np.float32),
            "mass": np.zeros((n_events, self.max_objects), dtype=np.float32),
            "n_trk": np.zeros((n_events, self.max_objects), dtype=np.float32),
            "mask": np.zeros((n_events, self.max_objects), dtype=bool),
            "n": np.full(n_events, 2, dtype=np.int32),
        }
        common_jets["p"][:, :2] = jet_p_val
        common_jets["eta"][:, :2] = jet_eta_val
        common_jets["phi"][:, :2] = jet_phi_val
        common_jets["mass"][:, :2] = jet_mass_val
        common_jets["n_trk"][:, :2] = jet_n_trk_val
        common_jets["mask"][:, :2] = True

        # --- 3. common/tracks (charged PF candidates from both jets in the event) ---
        is_charged = arrays["pfcand_charge"] != 0
        trk_p = arrays["pfcand_p"][is_charged]
        trk_theta = arrays["pfcand_theta"][is_charged]
        trk_pt = trk_p * np.sin(trk_theta)
        trk_theta_clipped = np.minimum(np.maximum(trk_theta, 1e-7), np.pi - 1e-7)
        trk_eta = -np.log(np.tan(trk_theta_clipped / 2.0))
        trk_phi = arrays["pfcand_phi"][is_charged]
        trk_d0 = arrays["pfcand_dxy"][is_charged]
        trk_z0 = arrays["pfcand_dz"][is_charged]

        def pair_and_pad_tracks(ak_arr):
            paired = ak.concatenate([ak_arr[0::2], ak_arr[1::2]], axis=1)
            padded = ak.pad_none(paired, self.max_tracks, axis=1, clip=True)
            return ak.to_numpy(ak.fill_none(padded, 0.0)).astype(np.float32)

        paired_trk_pt = ak.concatenate([trk_pt[0::2], trk_pt[1::2]], axis=1)
        trk_counts = ak.to_numpy(ak.num(paired_trk_pt, axis=1)).astype(np.int32)
        trk_mask = np.arange(self.max_tracks)[None, :] < np.minimum(trk_counts, self.max_tracks)[:, None]

        common_tracks = {
            "pt": pair_and_pad_tracks(trk_pt),
            "eta": pair_and_pad_tracks(trk_eta),
            "phi": pair_and_pad_tracks(trk_phi),
            "d0": pair_and_pad_tracks(trk_d0),
            "z0": pair_and_pad_tracks(trk_z0),
            "mask": trk_mask,
            "n": trk_counts,
        }

        # --- 4. fcc/jets (labels + constituent_offsets for jet 0 and jet 1) ---
        label_stack = np.stack([ak.to_numpy(arrays[b]) for b in LABEL_BRANCHES], axis=1)
        fcc_labels = np.zeros((n_events, self.max_objects), dtype=np.int32)
        fcc_labels[:, :2] = np.argmax(label_stack, axis=1).reshape(n_events, 2).astype(np.int32)

        n_const = ak.to_numpy(ak.num(arrays["pfcand_e"], axis=1)).astype(np.int64)
        ends = np.cumsum(n_const).reshape(n_events, 2)
        starts = ends - n_const.reshape(n_events, 2)

        # Shape: [n_events, max_objects + 1]
        # Jet 0: [offsets[i, 0] : offsets[i, 1]]
        # Jet 1: [offsets[i, 1] : offsets[i, 2]]
        # Empty slots 2..19: [ends[i, 1] : ends[i, 1]]
        constituent_offsets = np.repeat(ends[:, 1:2], self.max_objects + 1, axis=1)
        constituent_offsets[:, 0] = starts[:, 0]
        constituent_offsets[:, 1] = starts[:, 1]

        # --- 5. fcc/constituents/features (lossless flat [total_constituents, N_feat]) ---
        flat_cols = [
            ak.to_numpy(ak.flatten(arrays[feat])).astype(np.float32)
            for feat in PFCAND_FEATURES
        ]
        flat_features = np.stack(flat_cols, axis=1)

        # --- 6. Write HDF5 ---
        with h5py.File(output_file, "w") as f:
            f.attrs["sample_label"] = "simulation"

            common_grp = f.create_group("common")
            for sub, data_dict in [("event", common_event), ("jets", common_jets), ("tracks", common_tracks)]:
                g = common_grp.create_group(sub)
                for k, v in data_dict.items():
                    g.create_dataset(k, data=v)

            if pad_missing_common:
                for obj_name, fields in [
                    ("electrons", ["pt", "eta", "phi", "charge", "trk_iso03"]),
                    ("muons", ["pt", "eta", "phi", "charge", "trk_iso03"]),
                    ("taus", ["pt", "eta", "phi", "charge"]),
                    ("photons", ["pt", "eta", "phi", "trk_iso03"]),
                ]:
                    g = common_grp.create_group(obj_name)
                    for fld in fields:
                        g.create_dataset(fld, data=np.zeros((n_events, self.max_objects), dtype=np.float32))
                    g.create_dataset("mask", data=np.zeros((n_events, self.max_objects), dtype=bool))
                    g.create_dataset("n", data=np.zeros(n_events, dtype=np.int32))
                met_g = common_grp.create_group("met")
                for fld in ["pt", "phi", "sumet"]:
                    met_g.create_dataset(fld, data=np.zeros(n_events, dtype=np.float32))

            fcc_grp = f.create_group("fcc")
            fcc_jets = fcc_grp.create_group("jets")
            fcc_jets.create_dataset("labels", data=fcc_labels)
            fcc_jets.create_dataset("constituent_offsets", data=constituent_offsets)

            fcc_const = fcc_grp.create_group("constituents")
            fcc_const.create_dataset("features", data=flat_features)

            meta = f.create_group("metadata")
            meta.attrs["experiment"] = "FCC"
            meta.attrs["experiment_id"] = np.int64(self.experiment_id)
            meta.attrs["is_mc"] = np.True_
            meta.attrs["n_events"] = np.int64(n_events)
            meta.attrs["max_objects"] = np.int64(self.max_objects)
            meta.attrs["max_tracks"] = np.int64(self.max_tracks)
            meta.attrs["input_file"] = self.input_file
            meta.attrs["units_p"] = "GeV"
            meta.attrs["units_pt"] = "GeV"
            meta.attrs["units_energy"] = "GeV"
            meta.attrs["units_angles"] = "radians"
            meta.attrs["units_d0_z0"] = "mm"
            meta.attrs["constituent_feature_names"] = json.dumps(PFCAND_FEATURES)
            meta.attrs["label_names"] = json.dumps(LABEL_BRANCHES)

            schema = {
                "common": {
                    "event": list(common_event.keys()),
                    "jets": list(common_jets.keys()),
                    "tracks": list(common_tracks.keys()),
                },
                "fcc": {
                    "jets": ["labels", "constituent_offsets"],
                    "constituents": ["features"],
                },
            }
            meta.attrs["schema_json"] = json.dumps(schema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FCC ROOT jet ntuple to Treasure HDF5")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--tree", default="tree")
    parser.add_argument("--pad_missing_common", action="store_true",
                        help="Create zeroed/masked common/{electrons,muons,taus,photons,met} groups")
    args = parser.parse_args()

    FCCConverter(args.input, tree_name=args.tree).convert(
        args.output, pad_missing_common=args.pad_missing_common
    )
