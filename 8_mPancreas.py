"""Train & evaluate all 5 models on mPancreas (mouse — uses mouse Reactome GMT)."""
from run_all_models import run_all, MOUSE_GMT, MOUSE_GMT_URL

DATA_DIR = 'data/mPancreas'
CSV_PATH = 'results_mPancreas.csv'

if __name__ == '__main__':
    run_all(DATA_DIR, tag='mPancreas', gmt_path=MOUSE_GMT, gmt_url=MOUSE_GMT_URL,
            csv_path=CSV_PATH)
