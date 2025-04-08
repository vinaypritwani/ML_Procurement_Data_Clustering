import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score  
from feature_engineering import prepare_features  
from preprocessing import load_data  
import pickle

def find_optimal_clusters(features, max_k=50):
    """
    Determines the best number of clusters using silhouette score.
    Plots silhouette scores for k in [2..max_k].
    """
    scores = []
    ks = range(2, max_k + 1)
    
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        scores.append(score)
        print(f"Clusters: {k}, Silhouette Score: {score:.4f}")
    
    # Plot the silhouette scores
    plt.plot(ks, scores, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Optimal Number of Clusters")
    plt.show()
    
    return ks, scores

def run_clustering():
    """
    Loads procurement data, prepares features, finds an optimal k,
    fits KMeans, saves the model, and saves the clustered data.
    """
    # 1. Load Data
    df_procurement, _ = load_data()  # Adjust if load_data returns different objects
    
    # 2. Prepare Features
    features, _, _ = prepare_features(df_procurement)
    
    # 3. Find Optimal Clusters via Silhouette Score
    ks, _ = find_optimal_clusters(features, max_k=10)
    
    # For demonstration, pick an optimal k (e.g., 4). Adjust based on the plotted scores.
    optimal_k = 101
    
    # 4. Train KMeans with the chosen number of clusters
    model = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = model.fit_predict(features)
    
    # 5. Assign Cluster Labels to DataFrame
    df_procurement["Cluster"] = clusters
    print(df_procurement[["normalize_supplier_name", "invoice_description", "Cluster"]].head(10))
    
    # 6. Save the Model (Absolute Path to the 'models' folder)
    model_path = r"C:\Users\Vinay Pritwani\Desktop\procurement_ml_project\models\kmeans_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # 7. Save the Clustered Data (Absolute Path to the 'output' folder)
    csv_path = r"C:\Users\Vinay Pritwani\Desktop\procurement_ml_project\output\clustered_data7vinayj.csv"
    df_procurement.to_csv(csv_path, index=False)
    print(f"Clustered data saved to {csv_path}")

if __name__ == "__main__":
    run_clustering()
