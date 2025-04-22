print('starting!d(^_^o)')

import warnings
# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from mmgraphrag import MMGraphRAG
from time import time

pdf_path = './example_input/2020.acl-main.45.pdf'
WORKING_DIR = './example_output'
question = "How does the paper propose to calculate the coefficient \u03b1 for the Weighted Cross Entropy Loss?"

def index():
    rag = MMGraphRAG(
        working_dir=WORKING_DIR,
        input_mode=2
    )
    start = time()
    rag.index(pdf_path)
    print('success!ヾ(✿ﾟ▽ﾟ)ノ')
    print("indexing time:", time() - start)
def query():
    rag = MMGraphRAG(
        working_dir=WORKING_DIR,
        query_mode = True,
    )
    print(rag.query(question))

if __name__ == "__main__":
    index()
    query()