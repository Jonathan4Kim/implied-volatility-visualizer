from black_scholes import VolViz
import streamlit as st

st.title("Implied Volatility Visualizer")

st.button("Hello")

st.subheader("Dataframe")

visualizer = VolViz("MSFT")

fig, ax = visualizer.visualize_surface()

st.pyplot(fig=fig)


