from run_all_models import run_all, MOUSE_GMT, MOUSE_GMT_URL

DATA_DIR = 'data/mAtlas'
CSV_PATH = 'results_mAtlas.csv'

if __name__ == '__main__':
    run_all(DATA_DIR, tag='mAtlas', gmt_path=MOUSE_GMT, gmt_url=MOUSE_GMT_URL,
            csv_path=CSV_PATH)
