import pandas as pd
import streamlit as st
import pickle
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Fashion Recommendation System",
    page_icon="👗",
    layout="wide"
)

st.markdown("""
<style>

/* ===============================
   MAIN APP
=================================*/
.stApp{
    background: linear-gradient(135deg,#FFF8F4,#FFE8F0,#F8F1FF);
    color:#2D2D2D;
    font-family:'Segoe UI',sans-serif;
}

/* ===============================
   SIDEBAR
=================================*/
section[data-testid="stSidebar"]{
    background:#F9DCE7;
    border-right:2px solid #f3bfd0;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
    color:#7B1E52 !important;
    font-weight:800 !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label{
    color:#5A2D4D !important;
    font-weight:700 !important;
    font-size:16px !important;
}

/* ===============================
   TITLE
=================================*/
h1{
    color:#C2185B !important;
    font-size:52px !important;
    font-weight:900 !important;
}

h2,h3{
    color:#7B1E52 !important;
    font-weight:700;
}

p{
    color:#444 !important;
}

/* ===============================
   SEARCH BOX
=================================*/
.stTextInput input{

    background:#EAF7FF !important;
    color:#222 !important;

    border:2px solid #9DD9F3 !important;
    border-radius:12px;

    font-weight:600;
}

/* ===============================
   SELECT BOXES
=================================*/

/* Main select box */
div[data-baseweb="select"]{

    background:#E8FFF5 !important;

    border:2px solid #9BE7C4 !important;

    border-radius:12px;
}

/* Selected text inside dropdown */
div[data-baseweb="select"] span{

    color:#FFFFFF !important;

    font-weight:700 !important;

    font-size:15px !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg{
    fill:white !important;
}

/* Dropdown menu */
ul{
    background:#F5FFFB !important;
}

/* Options */
li{
    color:#222 !important;
    font-weight:600 !important;
}

/* Hover option */
li:hover{
    background:#D8FFF2 !important;
}

/* ===============================
   BUTTON
=================================*/
.stButton button{

    background:#E91E63 !important;

    color:white !important;

    border:none;

    border-radius:12px;

    font-weight:700;

    transition:0.3s;
}

.stButton button:hover{

    background:#C2185B !important;

    transform:scale(1.03);
}

/* ===============================
   SLIDER
=================================*/
.stSlider{

    color:#3AAFA9 !important;
}

/* ===============================
   ALERT BOXES
=================================*/
div[data-baseweb="notification"]{

    border-radius:12px;
}

/* ===============================
   IMAGES
=================================*/
img{

    border-radius:16px;

    box-shadow:0px 8px 25px rgba(0,0,0,0.15);
}

/* ===============================
   PRODUCT CARD TEXT
=================================*/
strong{
    color:#333 !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model Files
# -----------------------------
df = pickle.load(open("models/fashion_df.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

# Create TF-IDF matrix from tags
tfidf_matrix = tfidf.transform(df["tags"])

def recommend(product_name, top_n=5):

    indices = pd.Series(df.index, index=df["name"]).drop_duplicates()

    idx = indices[product_name]

    product_vector = tfidf_matrix[idx]

    similarity_scores = cosine_similarity(
        product_vector,
        tfidf_matrix
    ).flatten()

    sim_scores = list(enumerate(similarity_scores))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    product_indices = [i[0] for i in sim_scores]

    return df.loc[
        product_indices,
        [
            "name",
            "brand",
            "price",
            "category",
            "colour",
            "avg_rating",
            "img"
        ]
    ]
# Create the Outfit Generator Function
def generate_outfit(occasion, colour, budget):

    outfit = df.copy()

    # Filter by colour
    if colour != "Any":
        outfit = outfit[
            outfit["colour"].str.contains(colour, case=False, na=False)
        ]

    # Filter by budget
    outfit = outfit[outfit["price"] <= budget]

    # Filter according to occasion
    if occasion == "College":
        categories = ["Top", "Jeans", "Hoodie", "Sweatshirt"]

    elif occasion == "Office":
        categories = ["Top", "Trousers", "Shrug", "Cardigan"]

    elif occasion == "Party":
        categories = ["Dress", "Jumpsuit", "Skirt"]

    elif occasion == "Wedding":
        categories = ["Kurta", "Lehenga", "Saree", "Dupatta"]

    elif occasion == "Festival":
        categories = ["Kurta", "Saree", "Dupatta"]

    else:   # Casual
        categories = ["Top", "Jeans", "Trousers", "Dress"]

    outfit = outfit[outfit["category"].isin(categories)]

    if outfit.empty:
        return pd.DataFrame()

    return outfit.sample(min(5, len(outfit)))
    
# -----------------------------
# Title
# -----------------------------
# -----------------------------
# App Title
# -----------------------------
st.markdown("""
<h1>✨ StyleSense AI</h1>
<p style='font-size:22px;color:#6D4C5B;margin-top:-15px;'>
Your Personal AI Fashion Stylist 👗
</p>
""", unsafe_allow_html=True)

st.success("👋 Welcome! Search for your favorite fashion item or let AI create your perfect outfit.")
# Add a Sidebar
st.sidebar.header("🔍 Filters")

category_filter = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["category"].unique().tolist())
)

brand_filter = st.sidebar.selectbox(
    "Brand",
    ["All"] + sorted(df["brand"].unique().tolist())
)

budget = st.sidebar.slider(
    "Maximum Budget",
    min_value=int(df["price"].min()),
    max_value=int(df["price"].max()),
    value=5000
)
# Add Outfit Stylist Options
st.sidebar.markdown("---")
st.sidebar.header("✨ AI Outfit Stylist")

occasion = st.sidebar.selectbox(
    "Occasion",
    [
        "Casual",
        "College",
        "Office",
        "Party",
        "Wedding",
        "Festival"
    ]
)

preferred_colour = st.sidebar.selectbox(
    "Preferred Colour",
    ["Any"] + sorted(df["colour"].dropna().unique().tolist())
)
# Filter the Dataset
filtered_df = df.copy()

if category_filter != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == category_filter
    ]

if brand_filter != "All":
    filtered_df = filtered_df[
        filtered_df["brand"] == brand_filter
    ]

filtered_df = filtered_df[
    filtered_df["price"] <= budget
]
# -----------------------------
# Product Selection
# -----------------------------
search_text = st.text_input(
    "🔍 Search Product",
    placeholder="Type product or brand name..."
)

if search_text:

    matching_products = filtered_df[
        filtered_df["name"].str.contains(search_text, case=False, na=False)
    ]

    if len(matching_products) == 0:
        st.warning("No matching products found.")
        st.stop()

    selected_product = st.selectbox(
        "Select Matching Product",
        matching_products["name"].values
    )

else:
    st.info("Type something to search.")
    st.stop()

# Display Product Details Beautifully
selected_data = df[df["name"] == selected_product].iloc[0]

st.subheader("Selected Product")

col1, col2 = st.columns([1,2])

with col1:
    try:
        st.image(selected_data["img"], width=220)
    except:
        st.write("🖼️ Image not available")

with col2:
    st.write(f"### {selected_data['name']}")
    st.write(f"🏷 Brand: {selected_data['brand']}")
    st.write(f"💰 Price: ₹{int(selected_data['price'])}")
    st.write(f"👗 Category: {selected_data['category']}")
    st.write(f"🎨 Colour: {selected_data['colour']}")
    st.write(f"👩 Gender: {selected_data['gender']}")
    st.write(f"⭐ Rating: {round(selected_data['avg_rating'], 1)}")


# Add a Recommend Button
if st.button("✨ Get Recommendations"):

    recommendations = recommend(selected_product)

    st.subheader("Top 5 Recommended Products")

    for i, row in recommendations.iterrows():

        st.markdown("---")

        col1, col2 = st.columns([1, 2])

        with col1:
            try:
                st.image(row["img"], width=180)
            except:
                st.write("🖼️ Image not available")

        with col2:
            st.write(f"### {row['name']}")
            st.write(f"🏷 Brand: {row['brand']}")
            st.write(f"💰 Price: ₹{int(row['price'])}")
            st.write(f"👗 Category: {row['category']}")
            st.write(f"🎨 Colour: {row['colour']}")

            if row["avg_rating"] > 0:
                st.write(f"⭐ Rating: {round(row['avg_rating'], 1)}")
            else:
                st.write("⭐ Rating: Not Rated")

st.markdown("---")

if st.button("👗 Generate Outfit"):

    outfit = generate_outfit(
        occasion,
        preferred_colour,
        budget
    )

    if len(outfit) == 0:
        st.warning("No outfit found!")
    else:
        st.subheader("✨ AI Suggested Outfit")

        for _, row in outfit.iterrows():

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(row["img"], width=150)

            with col2:
                st.write(f"### {row['name']}")
                st.write(f"🏷 Brand: {row['brand']}")
                st.write(f"💰 ₹{int(row['price'])}")
                st.write(f"👗 Category: {row['category']}")
                st.write(f"🎨 Colour: {row['colour']}")

                if row["avg_rating"] > 0:
                    st.write(f"⭐ Rating: {round(row['avg_rating'], 1)}")
                else:
                    st.write("⭐ Rating: Not Rated")

            st.markdown("---")

            st.markdown("---")

st.caption("✨ StyleSense AI | Built with Python, Streamlit & Machine Learning")