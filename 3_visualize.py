
from allen_brain.cell_data.cell_dataset import make_dataset
from allen_brain.cell_data.cell_vis import plot_pca, plot_class_distribution, plot_heatmap, plot_umap


SEED = 42
def main():
    train_ds = make_dataset('data/smartseq')
    
    train_ds = train_ds if train_ds is not None else make_dataset('data/10x')
    pca, X_pca = plot_pca(train_ds, seed=SEED, n_components=20, save_path='figures/fig_pca_20.png')
    plot_class_distribution(train_ds)
    plot_heatmap(train_ds, gene_names=train_ds.gene_names, n_genes=20, save_path='figures/fig_heatmap_20.png')
    plot_umap(train_ds, train_ds.X, max_cells=4000, seed=SEED, save_path='figures/fig_umap.png')

if __name__ == '__main__':
    main()
    
    