"""Train & evaluate all 5 models on hPancreas (human — uses human Reactome GMT)."""
from run_all_models import run_all, HUMAN_GMT, HUMAN_GMT_URL

DATA_DIR = 'data/hPancreas'
CSV_PATH = 'results_hPancreas.csv'

if __name__ == '__main__':
    run_all(DATA_DIR, tag='hPancreas', gmt_path=HUMAN_GMT, gmt_url=HUMAN_GMT_URL,
            csv_path=CSV_PATH)
