# data = []
# for root, dirs, files in os.walk(os.path.join(data_dir, "16")):
#     subject = str(int(os.path.basename(root))) if os.path.basename(root).isdigit() else root
#     for file in files:
#         if file.endswith('.amc'):
#             index = str(int(file.split(".")[0].split("_")[1]))
#             if subject not in metadata or index not in metadata[subject]:
#                 continue
#             motion_info = metadata[subject][index]
#             motion_data = to_tensor(parse_motion_file(os.path.join(root, file)))

#             motion_data = to_target_frame_rate(motion_data, motion_info.frame_rate, config.frame_rate, average=False)
#             if len(motion_data)  < config.block_size + 1:
#                 continue # not enough frames to make a block
#             idx = torch.randint(0, (len(motion_data) - (config.block_size + 1)) + 1, (max(1, len(motion_data) * 2 // config.block_size),))
#             data.extend([motion_data[i: i + config.block_size + 1] for i in idx])

# data = torch.stack(data)


# dataset = TensorDataset(X, y)



# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size 

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_loader, test_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False), DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)