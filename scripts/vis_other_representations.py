import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def vis_euler_angles(quats, euler_path):
    r = R.from_quat(quats)
    euler_angles = r.as_euler('zyx', degrees=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    file_number = len(quats)
    ax.view_init(elev=file_number / 2, azim=file_number / 2)
    plt.tight_layout()
    plt.savefig(euler_path + '/euler_angles_' + str(file_number).zfill(3) + '.png')
    plt.close()


def vis_rotation_vectors(quats, rotvec_path):
    r = R.from_quat(quats)
    rotation_vectors = r.as_rotvec()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = np.linspace(0, 1, len(quats))
    ax.scatter(rotation_vectors[:, 0], rotation_vectors[:, 1], rotation_vectors[:, 2], c=color)
    file_number = len(quats)
    ax.view_init(elev=file_number / 2, azim=file_number / 2)
    plt.tight_layout()
    plt.savefig(rotvec_path + '/rotation_vectors_' + str(file_number).zfill(3) + '.png')
    plt.close()


def vis_quaternions_3d(quats, quat3d_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(quats[:, 0], quats[:, 1], quats[:, 2], c=quats[:, 3])
    file_number = len(quats)
    ax.view_init(elev=file_number / 2, azim=file_number / 2)
    plt.tight_layout()
    plt.savefig(quat3d_path + '/quaternions_3d_' + str(file_number).zfill(3) + '.png')
    plt.close()


def create_gif(images_folder, gif_path, duration=20, loop=0):
    images = []
    for filename in sorted(os.listdir(images_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            file_path = os.path.join(images_folder, filename)
            img = Image.open(file_path)
            images.append(img)
    images = sorted(images, key=lambda x: x.filename)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )


def main():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    img_path = os.path.join(dir_path, 'images/renderings')
    gif_path = os.path.join(dir_path, 'images/gifs')
    euler_path = os.path.join(dir_path, 'images/gifs/euler_angles')
    rotvec_path = os.path.join(dir_path, 'images/gifs/rotation_vectors')
    quat3d_path = os.path.join(dir_path, 'images/gifs/quaternions_3d')

    mkdir(img_path)
    mkdir(euler_path)
    mkdir(rotvec_path)
    mkdir(quat3d_path)

    for i in tqdm(range(3, 134)):
        optimal_samples = i
        generate_uniform_quats(optimal_samples)

        data_path = os.path.join(dir_path, 'SO3_sequence')
        quaternions = np.loadtxt(data_path + '/data.qua')

        vis_euler_angles(quaternions, euler_path)
        vis_rotation_vectors(quaternions, rotvec_path)
        vis_quaternions_3d(quaternions, quat3d_path)

    create_gif(euler_path, gif_path + '/euler_angles.gif')
    create_gif(rotvec_path, gif_path + '/rotation_vectors.gif')
    create_gif(quat3d_path, gif_path + '/quaternions_3d.gif')


if __name__ == '__main__':
    main()
