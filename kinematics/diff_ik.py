import mujoco as mj
import numpy as np

def differential_ik(model, data, joint_ids, x_des, site_ids, lam = 0.05):
    ee_site_id = site_ids["ee_site"] 
    J_pos = np.zeros((3, model.nv))         # if exclude gripper-->3*7
    J_rot = np.zeros((3, model.nv))         # if exclude gripper-->3*7
    mj.mj_jacSite(model, data, J_pos, J_rot, ee_site_id)      #if exclude gripper--> 6*7
    J = J_pos[:,:7]
    x = data.site_xpos[ee_site_id]
    e = x_des - x
    dq = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3) ) @ e
    return dq

def orientation_error(R_curr, R_des):
    R_err = R_des.T @ R_curr
    return 0.5 * np.array([
        R_err[2,1] - R_err[1,2],
        R_err[0,2] - R_err[2,0],
        R_err[1, 0] - R_err[0,1]
    ])

def differential_ik_pose(model, data, ee_site_id, joint_ids, x_des, R_des, lam = 0.05):
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, J_pos, J_rot,ee_site_id)

    Jp = J_pos[:, :7]
    Jr = J_pos[: , :7]
    

    x_curr = data.site_xpos[ee_site_id]
    R_curr = data.site_xmat[ee_site_id].reshape(3,3)

    e_pos = x_des - x_curr
    e_rot = orientation_error(R_curr, R_des)

    e= np.concatenate([e_pos, e_rot])
    J = np.vstack([Jp,Jr])

    dq = J.T @ np.linalg.inv(J@J.T + lam * np.eye(6)) @ e   # Damped least squares

    return dq