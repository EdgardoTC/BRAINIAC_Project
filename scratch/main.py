import os

def main(gp1_path, gp2_path, preprocessed_dir):
    flag_file = os.path.join(preprocessed_dir, "preprocessing_done.flag")

    if os.path.exists(flag_file):
        print("Preprocessing already done. Loading existing data...")
        # Just load existing .npy paths and labels
        import joblib
        all_scans, all_labels = joblib.load(os.path.join(preprocessed_dir, 'scan_index.pkl'))
    else:
        print("Starting preprocessing...")
        from preprocessing import preprocess_and_save  # import here to avoid unused import warning
        gp1_scans, gp1_labels = preprocess_and_save(gp1_path, 1, preprocessed_dir)
        gp2_scans, gp2_labels = preprocess_and_save(gp2_path, 0, preprocessed_dir)
        all_scans = gp1_scans + gp2_scans
        all_labels = gp1_labels + gp2_labels
        import joblib
        joblib.dump((all_scans, all_labels), os.path.join(preprocessed_dir, 'scan_index.pkl'))
        # Create the flag file
        with open(flag_file, 'w') as f:
            f.write("Preprocessing complete.")

    from train_eval import train_and_evaluate
    train_and_evaluate(all_scans, all_labels)
