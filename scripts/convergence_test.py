import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcd_registration_o3d import PointCloudRegistration


def generate_uniform_quats(num_samples):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(os.path.join(dir_path, 'SO3_sequence'))
    cpp_command = ['./SO3_sequence']
    process = subprocess.Popen(cpp_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    num_samples = f'{num_samples}\n'
    process.stdin.write(num_samples)
    process.stdin.flush()

    example_file = 'seq.txt\n'
    process.stdin.write(example_file)
    process.stdin.flush()

    stdout, stderr = process.communicate()


def generate_n_random_quats(num_samples):
    quaternions = []
    for i in range(num_samples):
        quat = np.random.uniform(-1, 1, size=4)
        quat = quat / np.linalg.norm(quat)
        quaternions.append(quat)
    return np.array(quaternions)


def verify_convergence(rotmat_fix, rotmat_mov, transformation): #TODO: result_rotmat, ground_truth_rotmat are probably not correct
    result_rotmat = transformation[:3, :3]
    ground_truth_rotmat = np.dot(rotmat_fix, rotmat_mov.T)

    result_rotmat_rotvec = R.from_matrix(np.array(result_rotmat)).as_rotvec()
    ground_truth_rotmat_rotvec = R.from_matrix(np.array(ground_truth_rotmat)).as_rotvec()
    norm_result_rotmat_rotvec = np.linalg.norm(result_rotmat_rotvec)
    norm_ground_truth_rotmat_rotvec = np.linalg.norm(ground_truth_rotmat_rotvec)
    cos_similarity = np.dot(result_rotmat_rotvec, ground_truth_rotmat_rotvec) / (
        norm_result_rotmat_rotvec * norm_ground_truth_rotmat_rotvec
    ) if norm_result_rotmat_rotvec * norm_ground_truth_rotmat_rotvec != 0 else 0
    convergence = True if cos_similarity > 0.97 else False
    print("Cosine similarity: {}".format(cos_similarity))
    print("Convergence: {}".format(convergence))
    return convergence


def main():
    pcd_reg = PointCloudRegistration()

    optimal_samples = 133
    K = 1
    generate_uniform_quats(optimal_samples)

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(dir_path, 'SO3_sequence')
    quaternions = np.loadtxt(data_path + '/data.qua')

    model_path = dir_path + '/models/102_valve_model.obj'
    points = o3d.io.read_triangle_mesh(model_path).vertices
    points = np.array(points) / 1000

    pcl_fix = o3d.geometry.PointCloud()
    pcl_fix.points = o3d.utility.Vector3dVector(points)
    pcl_fix.paint_uniform_color([0, 0.651, 0.929])
    
    pcl_mov = o3d.geometry.PointCloud()
    pcl_mov.points = o3d.utility.Vector3dVector(points)
    pcl_mov.paint_uniform_color([1, 0.706, 0])
    
    n_random_quats = generate_n_random_quats(200)

    all_convergence = []

    for i, random_quat in tqdm(enumerate(n_random_quats)):
        random_r = R.from_quat(random_quat)
        rotmat_fix = random_r.as_matrix()
        pcl_fix.rotate(rotmat_fix, center=(0, 0, 0))

        convergences = []

        for optimal_quat in tqdm(quaternions, leave=False):
            optimal_r = R.from_quat(optimal_quat)
            rotmat_mov = optimal_r.as_matrix()

            pcl_mov.rotate(rotmat_mov, center=(0, 0, 0))

            # pcd_reg.draw_registration_result(pcl_mov, pcl_fix, result=None, window_name="Before registration")
            result_local_refinement, _ = pcd_reg.execute_multi_scale_ICP_registration(
                pcl_mov, pcl_fix, pcd_reg.voxel_sizes, np.identity(4)
            )

            transformation = result_local_refinement.transformation.cpu().numpy()

            pcl_mov.transform(transformation)
            pcd_reg.draw_registration_result(pcl_mov, pcl_fix, result=None, window_name="After registration")

            convergence = verify_convergence(rotmat_fix, rotmat_mov, transformation)
            convergences.append(convergence)

        if np.any(convergences):
            all_convergence.append(True)
        else:
            all_convergence.append(False)

    print("Convergence rate: {}".format(np.sum(all_convergence) / len(all_convergence)))


if __name__ == '__main__':
    main()
