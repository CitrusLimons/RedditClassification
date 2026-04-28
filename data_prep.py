import pandas as pd
import re
import html

# Load input files
posts = pd.read_csv(r"D:\downloads\InformationRet\rspct.tsv", sep="\t")
subreddits = pd.read_csv(r"D:\downloads\InformationRet\subreddit_info.csv")

print("rspct.tsv shape:", posts.shape)
print("subreddit_info.csv shape:", subreddits.shape)

# Join subreddit metadata onto the posts table
subreddits = subreddits.set_index("subreddit")
df = posts.join(subreddits, on="subreddit")

# Rename columns used later
df = df.rename(columns={"selftext": "text", "title": "title"})

# Keep only columns needed for processing
keep_cols = ["title", "text", "subreddit"]
if "category_1" in df.columns:
    keep_cols.append("category_1")

keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()

# Map subreddits into broader categories
subreddit_to_category = {
    # Gaming
    "gaming": "Gaming & Esports",
    "pcgaming": "Gaming & Esports",
    "wow": "Gaming & Esports",
    "leagueoflegends": "Gaming & Esports",
    "minecraft": "Gaming & Esports",
    "smashbros": "Gaming & Esports",

    # Tech
    "technology": "Tech & Gadgets",
    "gadgets": "Tech & Gadgets",
    "programming": "Tech & Gadgets",
    "buildapc": "Tech & Gadgets",
    "intel": "Tech & Gadgets",
    "talesfromtechsupport": "Tech & Gadgets",

    # Food
    "food": "Food & Cooking",
    "recipes": "Food & Cooking",
    "cooking": "Food & Cooking",

    # Movies and TV
    "movies": "Movies & TV Shows",
    "television": "Movies & TV Shows",

    # Music
    "music": "Music",
    "listentothis": "Music",

    # News and politics
    "news": "Current Affairs",
    "worldnews": "Current Affairs",
    "politics": "Current Affairs",

    # Science
    "science": "Science & Space",
    "askscience": "Science & Space",
    "space": "Science & Space",

    # Humor
    "funny": "Comedy & Jokes",
    "jokes": "Comedy & Jokes",
    "dadjokes": "Comedy & Jokes",

    # Books
    "books": "Books & Literature",
    "suggestmeabook": "Books & Literature",

    # Art
    "art": "Art & Design",
    "photoshopbattles": "Art & Design",

    # Fitness
    "fitness": "Fitness & Wellness",

    # Relationships
    "relationships": "Relationships & Dating",
    "dating_advice": "Relationships & Dating",

    # Lifestyle
    "aww": "Lifestyle & Daily Life",
    "pics": "Lifestyle & Daily Life",
    "todayilearned": "Lifestyle & Daily Life",
    "explainlikeimfive": "Lifestyle & Daily Life",
    "showerthoughts": "Lifestyle & Daily Life",
    "tifu": "Lifestyle & Daily Life",
    "lifeprotips": "Lifestyle & Daily Life",

    # Hobbies
    "diy": "Hobbies",
    "harley": "Hobbies",

    # Other
    "other": "Other",
}

# Fill missing mappings with Other
df["category"] = df["subreddit"].map(subreddit_to_category).fillna("Other")

# Use category_1 if it exists
if "category_1" in df.columns:
    df["category"] = df["category_1"].fillna(df["category"])

# Combine title and body text
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["full_text"] = (df["title"] + " " + df["text"]).str.strip()

# Clean text
def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["full_text"] = df["full_text"].apply(clean_text)

# Remove incomplete or very short rows
df = df.dropna(subset=["subreddit", "category", "full_text"])
df = df[df["full_text"].str.len() > 10]

print(f"Rows before category filter: {len(df):,}")
print("Top categories before filtering:")
print(df["category"].value_counts().head(15))

# Keep the top 40 categories
top_categories = df["category"].value_counts().head(40).index
df = df[df["category"].isin(top_categories)].copy()

print(f"\nRows after top-40 filter: {len(df):,}")
print("Final category distribution:")
print(df["category"].value_counts())

# Save output file for modeling
final_df = df[["full_text", "subreddit", "category"]].copy()
final_df.columns = ["text", "original_subreddit", "category"]
final_df.to_csv(r"D:\downloads\InformationRet\reddit_merged_categories.csv", index=False)

print("\nSaved: D:\\downloads\\InformationRet\\reddit_merged_categories.csv")
print("Dataset is ready for modeling")

# Quick check
df = pd.read_csv(r"D:\downloads\InformationRet\reddit_merged_categories.csv")
print("Number of categories:", df["category"].nunique())
print("Final shape:", df.shape)
