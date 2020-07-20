import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import autoscale, SNV
from src.pca import pca
# import altair as alt
plt.style.use("zs")

# title
st.title("Demo")

# body
# show data
st.write("# 1.Read a csv file of spectra data")
data = pd.read_csv("./data/pure_spec.csv", header=None)
st.write(f"### The shape of the spectra data is `{data.shape}`")
st.write("The first several rows of the spectra data is shwon below")
st.dataframe(data.head())
spectra = data.values

# show the original spectra graph
st.write("# 2.The original spectra")


def plot_spectra(spectra, step_sample=1, step_variable=1):
    variable_list = [i for i in range(0, spectra.shape[1], step_variable)]
    sample_list = [j for j in range(0, spectra.shape[0], step_sample)]
    fig = plt.figure()
    for i in sample_list:
        plt.plot(spectra[i, variable_list])
    return fig


step_sample_original = st.slider(label="sample step",
                                 min_value=1,
                                 max_value=spectra.shape[0],
                                 value=3,
                                 step=1)
step_variable_original = st.slider(label="variable step",
                                   min_value=1,
                                   max_value=100,
                                   value=10,
                                   step=1)
st.write(
    plot_spectra(spectra,
                 step_sample=step_sample_original,
                 step_variable=step_variable_original))

# select preprocessing techniques
st.write("# 3.Choose one preprocessing technique")
pre_tech = st.selectbox("", ("autoscale", "SNV", "None"))
if pre_tech == "autoscale":
    pre_spectra = autoscale(spectra)
elif pre_tech == "SNV":
    pre_spectra = SNV(spectra)
elif pre_tech == "None":
    pre_spectra = spectra
else:
    print("please choose proper preprocessing techniques")

step_sample_pre = st.slider(label="sample step after preprocessing",
                            min_value=1,
                            max_value=pre_spectra.shape[0],
                            value=3,
                            step=1)
step_variable_pre = st.slider(label="variable step afer preprocessing",
                              min_value=1,
                              max_value=100,
                              value=10,
                              step=1)
st.write(
    plot_spectra(pre_spectra,
                 step_sample=step_sample_pre,
                 step_variable=step_variable_pre))

# PCA
# select n_components
st.write("# 4. Principal component analysis")
pca_model_demo = pca(pre_spectra, n_components=30)
variance_ratio_ = pca_model_demo.pca_model.explained_variance_ratio_
variance_ratio_cumsum = pd.DataFrame(
    np.cumsum(variance_ratio_),
    columns=["cumulative explained variance ratio"])
st.bar_chart(variance_ratio_cumsum)

# build model
st.write("**Choose component number**")
n_component = st.number_input("", value=3)
pca_model = pca(pre_spectra, n_components=n_component)

axis_x = st.number_input("choose the PC number of X axis (zero-index)",
                         value=0)
axis_y = st.number_input("choose the PC number of Y axis (zero-index)",
                         value=1)
pca_model.plot_2d_score(axis_x=axis_x, axis_y=axis_y)
st.pyplot()
