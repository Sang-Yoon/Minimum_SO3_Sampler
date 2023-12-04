import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import pandas as pd
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from tqdm import tqdm


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


def main():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(dir_path, 'SO3_sequence')
    angle_data = []
    for i in tqdm(range(3, 150)):
        generate_uniform_quats(i)
        quaternions = np.loadtxt(data_path + '/data.qua')
    
        rotvecs = []
        for j in range(len(quaternions)):
            rotvecs.append(R.from_quat(quaternions[j]).as_rotvec())

        rotvecs = np.array(rotvecs)
        
        kdtree = KDTree(rotvecs)
        K = 1
        _, idx = kdtree.query(rotvecs, k=K + 1)

        angles = []
        for k in range(len(idx)):
            query_rotvec = rotvecs[k]
            knn_rotvecs = rotvecs[idx[k]][1:]
            for neighbor_rotvec in knn_rotvecs:
                dot_product = np.dot(query_rotvec, neighbor_rotvec)
                cos_angle = np.clip(dot_product / (np.linalg.norm(query_rotvec) * np.linalg.norm(neighbor_rotvec)), -1.0, 1.0)
                angles.append(np.arccos(cos_angle))

        angles = np.array(angles)
        angles = np.rad2deg(angles)

        angle_data.append([i, angles])

    df = pd.DataFrame(angle_data, columns=['Number of samples', 'Angles'])
    df = df.explode('Angles')
    df['Angles'] = df['Angles'].astype(float)
    df['Number of samples'] = df['Number of samples'].astype(int)

    num_samples = df['Number of samples'].unique()
    angles_by_num_samples = []
    for i in num_samples:
        angles_by_num_samples.append([i, df[df['Number of samples'] == i]['Angles'].values])

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x='Number of samples', y='Angles', data=df)
    plt.axhline(y=20, color='r', linestyle='dashed')
    plt.ylim(0, 80)
    ax.tick_params(axis='x', rotation=90)
    img_path = os.path.join(dir_path, 'images')
    plt.savefig(os.path.join(img_path, 'optimal_pose_samples.png'), bbox_inches='tight', dpi=300)

    for angles in angles_by_num_samples:
        q1 = np.quantile(angles[1], 0.25)
        q3 = np.quantile(angles[1], 0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        angles[1] = angles[1][(angles[1] > lower_bound) & (angles[1] < upper_bound)]
        df = df[(df['Number of samples'] != angles[0]) | ((df['Number of samples'] == angles[0]) & (df['Angles'] > lower_bound) & (df['Angles'] < upper_bound))]

    optimal_num_samples = None
    for angles in angles_by_num_samples:
        if len(angles[1]) == 0:
            continue
        elif np.all(angles[1] < 20):
            print('All angles are less than 20 degrees for {} samples'.format(angles[0]))
            optimal_num_samples = angles[0]
            break

    print('Optimal number of samples: {}'.format(optimal_num_samples))

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x='Number of samples', y='Angles', data=df)
    plt.axvline(x=optimal_num_samples - 3.5, color='b', linestyle='solid')
    plt.axhline(y=20, color='r', linestyle='dashed')
    plt.ylim(0, 80)
    ax.tick_params(axis='x', rotation=90)
    # plt.show()
    plt.savefig(os.path.join(img_path, 'optimal_pose_samples_remove_outliers.png'), bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
