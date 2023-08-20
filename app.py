import pandas as pd
import numpy as np
import streamlit as st
from faker import Faker
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

# Load your datasets
user_df = pd.read_csv("user_dataset1.csv")  # Replace with your actual file path
product_df = pd.read_csv("product_dataset1.csv")  # Replace with your actual file path
interaction_df = pd.read_csv("interaction_dataset1.csv")  # Replace with your actual file path

# Preprocessing: Create user and product dictionaries for mapping IDs to indices
user_dict = {user_id: index for index, user_id in enumerate(user_df["user_id"])}
product_dict = {product_id: index for index, product_id in enumerate(product_df["product_id"])}

# Preprocessing: Create user and product interaction matrices
user_indices = [user_dict[user_id] for user_id in interaction_df["user_id"]]
product_indices = [product_dict[product_id] for product_id in interaction_df["product_id"]]
interaction_values = np.ones(len(user_indices))

interactions_matrix = csr_matrix((interaction_values, (user_indices, product_indices)),
                                 shape=(len(user_dict), len(product_dict)))

# Preprocessing: Create user preference vectors using TF-IDF for content-based filtering
vectorizer = TfidfVectorizer()
product_descriptions = product_df["description"].fillna("")
product_description_vectors = vectorizer.fit_transform(product_descriptions)
user_preferences_matrix = interactions_matrix.dot(product_description_vectors)

# Initialize the BPR model for collaborative filtering
bpr_model = BayesianPersonalizedRanking(factors=50, iterations=50)
bpr_model.fit(interactions_matrix)

# Streamlit UI
st.title("Hybrid Product Recommendation System")

# Streamlit UI to take user input
num_users = st.slider("Number of Users", min_value=10, max_value=100, value=50)
num_products = st.slider("Number of Products", min_value=20, max_value=200, value=100)
num_interactions = st.slider("Number of Interactions", min_value=100, max_value=1000, value=500)

if st.button("Generate Recommendations"):
    # User ID for which recommendations are needed
    target_user_id = user_df.loc[5, 'user_id']
    user_index = user_dict.get(target_user_id)

    if user_index is None:
        st.write("Target user not found in the interaction matrix.")
    elif user_index >= len(bpr_model.user_factors):
        st.write("User index is out of bounds for collaborative filtering model.")
    else:
        # Generate content-based recommendations
        st.markdown(target_user_id)
        user_preference_vector = user_preferences_matrix[user_index]
        product_scores = product_description_vectors.dot(user_preference_vector.T)
        recommended_product_indices_content = np.argsort(product_scores.A.flatten())[::-1][:10]
        recommended_product_ids_content = [list(product_dict.keys())[list(product_dict.values()).index(product_index)] for product_index in recommended_product_indices_content]
        
        # Generate collaborative filtering recommendations
        user_latent_factors = bpr_model.user_factors[user_index]
        user_latent_factors_csr = csr_matrix(user_latent_factors.reshape(1, -1))
        recommended_product_indices_collab, _ = bpr_model.recommend(user_index, user_latent_factors_csr, N=10)
        recommended_product_ids_collab = [list(product_dict.keys())[list(product_dict.values()).index(product_index)] for product_index in recommended_product_indices_collab]

        # Merge content-based and collaborative recommendations
        recommended_product_ids = list(set(recommended_product_ids_content) | set(recommended_product_ids_collab))
        recommended_products_df = product_df[product_df["product_id"].isin(recommended_product_ids)]
        
        st.subheader("Recommended Products for User")
        st.dataframe(recommended_products_df)
