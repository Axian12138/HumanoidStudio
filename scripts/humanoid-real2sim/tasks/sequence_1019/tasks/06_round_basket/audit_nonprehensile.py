"""Audit continuous foot/shin and forearm basket contact at 50 Hz plus half frames."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from robosimstudio.humanoid_real2sim.paths import example_dir

from audit_hoi import geometry


HERE = example_dir(__file__)
OUT = HERE / "hoi_v3_nonprehensile"
MODEL = Path("/home/galbot/workspace/ScaleTrack/source/scaletrack/scaletrack/assets/robots/"
             "g1_29dof_dex3/g1_29dof_dex3_offsethand.xml")
ORDER = HERE.parents[1] / "retarget/ground_support_27_end/sidecars/motion_order.json"
URDF = Path("/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/body.urdf")
FOOT = ("left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link")
ARM = ("right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
       "right_wrist_yaw_link", "right_hand_thumb_0_link", "right_hand_middle_0_link",
       "right_hand_index_0_link")


def meshes(model):
    result = {}
    for name in FOOT + ARM:
        bid = model.body(name).id
        local = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != bid or model.geom_group[geom] == 3 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[geom]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            local.append(Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).apply(v) + model.geom_pos[geom])
        if local:
            vertices = np.concatenate(local)
            result[name] = (bid, vertices[ConvexHull(vertices).vertices])
    return result


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    names = json.loads(ORDER.read_text())["robot_joint_names"]
    addresses = np.asarray([model.joint(name).qposadr[0] for name in names])
    parts = geometry(HERE / "round_laundry_basket_collision.obj")
    body_meshes = meshes(model)
    with np.load(OUT / "motion.npz") as motion, np.load(OUT / "body_prior_69_76.npz") as prior, \
         np.load(OUT / "object_estimate.npz") as obj, np.load(OUT / "human_motion_world.npz") as human, \
         np.load(OUT / "human_69_76.npz") as source_human:
        q = motion["joint_pos"].astype(float)
        old = prior["joint_pos"].astype(float)
        position = obj["position_w"].astype(float)
        aligned = bool(np.allclose(human["positions_w"], source_human["keypoints_w"], atol=1e-6))
        times = sorted(set(np.arange(350, dtype=float).tolist() +
                           (np.arange(round((69.65-69)*50), round((73.9-69)*50)) + 0.5).tolist()))
        contact = {name: [] for name in FOOT + ARM}
        collisions = []
        alignment = {"foot": [], "arm": []}
        foot_force_bracket = []
        knee_floor = []
        foot_floor = []
        for sample in times:
            i = int(sample)
            a = sample - i
            j = min(i + 1, len(q) - 1)
            data.qpos[:3] = (1-a)*motion["body_pos_w"][i, 0] + a*motion["body_pos_w"][j, 0]
            qa = Rotation.from_quat(motion["body_quat_w"][i, 0, [1,2,3,0]])
            qb = Rotation.from_quat(motion["body_quat_w"][j, 0, [1,2,3,0]])
            interp = qa * Rotation.from_rotvec((qa.inv()*qb).as_rotvec()*a)
            data.qpos[3:7] = interp.as_quat()[[3,0,1,2]]
            data.qpos[addresses] = (1-a)*q[i] + a*q[j]
            mujoco.mj_forward(model, data)
            center = (1-a)*position[i] + a*position[j]
            travel = position[min(i+3,len(q)-1),:2] - position[max(i-3,0),:2]
            t = 69 + sample/50
            phase = "foot" if 69.75 <= t <= 70.3 else "arm" if 72.0 <= t <= 73.8 else None
            contact_vectors = {}
            for name, (bid, local) in body_meshes.items():
                world = local @ data.xmat[bid].reshape(3,3).T + data.xpos[bid]
                if name == "left_knee_link" and 70.9 <= t <= 73.8:
                    knee_floor.append(float(world[:,2].min()))
                if name == "left_ankle_roll_link" and 69.65 <= t <= 70.5:
                    foot_floor.append(float(world[:,2].min()))
                per_body = []
                local_points = world - center
                for part, lower, upper, eq, _ in parts:
                    nearby = np.all((local_points >= lower-.025) & (local_points <= upper+.025), axis=1)
                    if not nearby.any():
                        continue
                    signed = np.max(local_points[nearby] @ eq[:,:3].T + eq[:,3], axis=1)
                    near_world = world[nearby]
                    index = np.argmin(np.abs(signed))
                    per_body.append((float(signed[index]), near_world[index], part))
                    if np.min(signed) < -0.001:
                        collisions.append({"time_s": round(t,3), "body": name, "part": part,
                                           "depth_m": float(-np.min(signed))})
                if per_body:
                    signed, point, part = min(per_body, key=lambda item: abs(item[0]))
                    contact[name].append((t, signed, part))
                    if phase == "foot" and name in ("left_knee_link", "left_ankle_roll_link"):
                        contact_vectors[name] = (center[:2]-point[:2], abs(signed))
                    if phase and name.startswith("left_") == (phase == "foot") and np.linalg.norm(travel) > 1e-5:
                        force = center[:2] - point[:2]
                        norm = np.linalg.norm(force)
                        if norm > 1e-6:
                            alignment[phase].append((t, name, float(force @ travel / (norm*np.linalg.norm(travel))), abs(signed)))
            if 69.8 <= t <= 70.2 and all(name in contact_vectors for name in ("left_knee_link", "left_ankle_roll_link")) and np.linalg.norm(travel)>1e-5:
                shin, shin_gap = contact_vectors["left_knee_link"]
                foot, foot_gap = contact_vectors["left_ankle_roll_link"]
                if shin_gap <= .01 and foot_gap <= .01:
                    cross_shin = shin[0]*travel[1]-shin[1]*travel[0]
                    cross_foot = foot[0]*travel[1]-foot[1]*travel[0]
                    foot_force_bracket.append(cross_shin*cross_foot <= 0)

    velocities = {j.get("name"):float(j.find("limit").get("velocity",20))
                  for j in ET.parse(URDF).findall("joint") if j.find("limit") is not None}
    speed = np.asarray([velocities.get(name,20.0) for name in names])
    limits = {}
    for col, name in enumerate(names):
        joint = model.joint(name)
        if joint.limited:
            exceed = max(0, joint.range[0]-q[:,col].min(), q[:,col].max()-joint.range[1])
            if exceed > 1e-4:
                limits[name] = float(exceed)
    changed = [name for col,name in enumerate(names) if np.max(abs(q[:,col]-old[:,col])) > 1e-6]
    def window(name, lo, hi):
        values = [(t,gap) for t,gap,_ in contact[name] if lo <= t <= hi]
        return {"sample_count": len(values),
                "min_abs_gap_m": min((abs(gap) for _,gap in values), default=None),
                "max_abs_gap_m": max((abs(gap) for _,gap in values), default=None),
                "fraction_within_10mm": float(np.mean([abs(gap)<=.01 for _,gap in values])) if values else None}
    report = {
        "status": "kinematic_geometry_audit_not_physical_force_measurement",
        "source_human_exact_69_76_alignment": aligned,
        "changed_joints": changed,
        "only_left_leg_and_right_arm_changed": all(name.startswith(("left_hip_", "left_knee_", "left_ankle_", "right_shoulder_", "right_elbow_", "right_wrist_")) for name in changed),
        "dex3_unchanged": all("hand" not in name for name in changed),
        "seam_joint_max_rad": float(max(np.max(abs(q[0]-old[0])),np.max(abs(q[-1]-old[-1])))),
        "max_joint_speed_limit_ratio": float(np.max(abs(np.diff(q,axis=0))*50/speed)),
        "joint_limit_excess_rad": limits,
        "left_knee_floor_min_m": min(knee_floor),
        "left_foot_floor_min_m": min(foot_floor),
        "foot_shin_contact": {name:window(name,69.75,70.3) for name in FOOT},
        "moving_foot_shin_contact": {name:window(name,69.8,70.2) for name in FOOT},
        "right_forearm_contact": {name:window(name,72.0,73.8) for name in ARM},
        "force_alignment": {phase:{"min_cosine":min((x[2] for x in items if x[3]<.015),default=None),
                                    "mean_cosine":float(np.mean([x[2] for x in items if x[3]<.015])) if any(x[3]<.015 for x in items) else None}
                            for phase,items in alignment.items()},
        "moving_force_ray_between_foot_and_shin_fraction": float(np.mean(foot_force_bracket)) if foot_force_bracket else None,
        "moving_force_ray_bracket_sample_count":len(foot_force_bracket),
        "sampled_collision_count": len(collisions),
        "sampled_max_penetration_m": max((x["depth_m"] for x in collisions),default=0),
        "deepest_collisions": sorted(collisions,key=lambda x:x["depth_m"],reverse=True)[:20],
        "sample_count":len(times),
        "limits": "robot mesh convex vertices vs 65 original basket convex hulls; 50Hz plus half frames in 69.65–73.9 s",
    }
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:report[key] for key in (
        "source_human_exact_69_76_alignment","only_left_leg_and_right_arm_changed","dex3_unchanged",
        "seam_joint_max_rad","max_joint_speed_limit_ratio","joint_limit_excess_rad",
        "left_knee_floor_min_m","left_foot_floor_min_m","foot_shin_contact","moving_foot_shin_contact","right_forearm_contact",
        "force_alignment","moving_force_ray_between_foot_and_shin_fraction",
        "sampled_collision_count","sampled_max_penetration_m")},indent=2))


if __name__ == "__main__":
    main()
