import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import open3d as o3d
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


def make_single_image(imgs):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    img_path = os.path.join(dir_path, 'images')
    rendering_img_path = os.path.join(img_path, 'renderings')
    imgs = [os.path.join(rendering_img_path, img) for img in imgs]

    images = [plt.imread(img) for img in imgs]
    len_images = len(images)
    h = 10
    w = len_images // h + 1
    fig, ax = plt.subplots(w, h)
    for i in tqdm(range(w)):
        for j in range(h):
            if i * h + j >= len_images:
                ax[i, j].axis('off')
            else:
                ax[i, j].imshow(images[i * h + j])
                ax[i, j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(img_path, 'housing.png'), bbox_inches='tight', pad_inches=0)


def main():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    img_path = os.path.join(dir_path, 'images/renderings')
    mkdir(img_path)

    optimal_samples = 133
    generate_uniform_quats(optimal_samples)

    data_path = os.path.join(dir_path, 'SO3_sequence')
    quaternions = np.loadtxt(data_path + '/data.qua')

    model_path = dir_path + '/models/housing.obj'
    mesh = o3d.io.read_triangle_mesh(model_path)

    rendering_img_path = os.path.join(dir_path, 'images/renderings')
    for i, quat in tqdm(enumerate(quaternions), total=len(quaternions)):
        r = R.from_quat(quat)
        rot = r.as_matrix()
        mesh.rotate(rot, center=(0, 0, 0))
        mesh.translate([0, 0, 0])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.5, 0.5])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        image = vis.capture_screen_float_buffer(True)
        plt.imsave(f'{rendering_img_path}/{i}.png', np.asarray(image))
        vis.destroy_window()

    imgs = []
    for dirpath, dirnames, filenames in os.walk(rendering_img_path):
        for filename in filenames:
            imgs.append(filename)

    make_single_image(imgs)


if __name__ == '__main__':
    main()
