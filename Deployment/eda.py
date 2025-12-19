import streamlit as st
import os


def run():
    # Create title
    st.title("Rock-Paper-Scissors Image Classification App")
    st.write("This app uses a Deep Learning model to predict hand gestures forming rock, paper, or scissors.")

    # Create link for dataset
    st.write("The dataset used in this app is taken from Kaggle. It contains images of hands forming rock, paper, or scissors gestures.")
    st.markdown("[View Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data)")

    # Display EDA results
    st.subheader("Exploratory Data Analysis (EDA)")


    # --- Dynamic Image Path Logic ---
    # Look for local user location if exists, otherwise use relative src folder
    local_path = r'X:\FTDS\P2\GC7\src'
    if os.path.exists(local_path):
        base_src = local_path
    else:
        # For deployment (relative to this script)
        base_src = os.path.join(os.path.dirname(__file__), 'src')

    st.write("1. EDA 1: Image Visualization")
    st.image(os.path.join(base_src, "eda1.png"), caption="EDA 1 Result: Image visualization from Rock class", use_container_width=True)
    st.image(os.path.join(base_src, "eda1.2.png"), caption="EDA 1 Result: Image visualization from Paper class", use_container_width=True)
    st.image(os.path.join(base_src, "eda1.3.png"), caption="EDA 1 Result: Image visualization from Scissors class", use_container_width=True)
    st.write("In the dataset, each image has a green background. This is done because the color green is very sparse on the human body, so using a green background improves the model's performance in distinguishing hand movements effectively.")
    st.divider()

    st.write("2. EDA 2: Class Distribution")
    st.image(os.path.join(base_src, "eda2.png"), caption="EDA 2 Result: Class Distribution", use_container_width=True)  
    st.write("The chart above shows the distribution of images for each class (Rock, Paper, Scissors) in the dataset.")
    st.write("Number of scissors images: 750")
    st.write("Number of paper images: 712")
    st.write("Number of rock images: 726")
    st.write("The dataset uses an almost balanced number of images for each class, meaning no single class dominates. Because the data is balanced, there is no need for oversampling or undersampling.")
    st.divider()  

    st.write("3. EDA 3: Image Dimension Analysis")
    st.write("In EDA 3, we analyze the dimensions of the images used in the dataset.")
    st.write("Based on this analysis, images in each class have the same dimensions, which is 300x200 pixels. This is important to ensure that the model processes images with a consistent size. Since all data in the dataset has the same dimensions, no standardization is required.")
    st.divider()


    st.write("4. EDA 4: Color Type Analysis")
    st.image(os.path.join(base_src, "eda4.png"), caption="EDA 4 Result: Color Type Analysis", use_container_width=True)
    st.write("Based on this analysis, if the image is converted to green, the hand shape is not clearly visible. Therefore, the model should not prioritize weights on features in the green color channel.")
    st.divider()

    st.write("5. EDA 5: Object Position Analysis")
    st.image(os.path.join(base_src, "eda5.1.png"), caption="EDA 5 Result: Object Position Analysis - Rock Class", use_container_width=True)
    st.image(os.path.join(base_src, "eda5.2.png"), caption="EDA 5 Result: Object Position Analysis - Paper Class", use_container_width=True)
    st.image(os.path.join(base_src, "eda5.3.png"), caption="EDA 5 Result: Object Position Analysis - Scissors Class", use_container_width=True)
    st.write("Based on the images, the hand position is always similar. Therefore, new data (augmentation) should be created to vary the hand positions so the model can better recognize gestures in different locations.")
    st.divider()

    st.write("6. EDA 6: Pixel Intensity Analysis")
    st.image(os.path.join(base_src, "eda6.1.png"), caption="EDA 6 Result: Pixel Intensity Analysis - Rock Class", use_container_width=True)
    st.image(os.path.join(base_src, "eda6.2.png"), caption="EDA 6 Result: Pixel Intensity Analysis - Paper Class", use_container_width=True)
    st.image(os.path.join(base_src, "eda6.3.png"), caption="EDA 6 Result: Pixel Intensity Analysis - Scissors Class", use_container_width=True)
    st.write("Based on the grayscale color distributions, the dataset has good contrast values, which will make it easier for the model to detect hand shapes.")
 
if __name__ == "__main__":
    run()  