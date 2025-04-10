import color_classification as cc

def main():
    image_path = r"miniprojekt_3\Cropped and perspective corrected boards"
    label_path = r"labels_uden_kroner.csv"

    classifier = cc.Tile_Classifier()
    classifier.run_pipeline(image_path, label_path)


if __name__ == "__main__":
    main()