import color_classification as cc
color_classification = cc.Tile_Classifier()


def main():
        image_path = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"
        label_path = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\labels_uden_kroner.csv"

        # Create an instance of the class
        classifier = cc.Tile_Classifier()
        classifier.run_pipeline(image_path, label_path)


if __name__ == "__main__":
    main()