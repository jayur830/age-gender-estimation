import h5py

if __name__ == '__main__':
    with h5py.File("D:/Dataset/face_dataset/test.hdf5", "r") as f:
        x_data = f["x_data"]
        y_age = f["y_age"]
        print(x_data.shape)
        print(y_age.shape)
