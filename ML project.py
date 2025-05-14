# Step 1: Import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 2: Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Visualize the result
pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])  # Create a DataFrame with PCA components
pca_df['species'] = y  # Add species labels for color coding

# Plot the data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="species", palette="viridis", s=100, alpha=0.7)
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
