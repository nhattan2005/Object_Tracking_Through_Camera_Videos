# app.py (bổ sung endpoint mới)

from flask import Flask, request, jsonify
from my_tracker import extract_and_track
from reid_indexer import index_reid
from search_reid import search_text_reid
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

@app.route("/process_video_track", methods=["POST"])
def process_video_track():
    video_path = request.json.get("video_path")
    metadata = extract_and_track(video_path)
    print(f"[DEBUG] Extracted {len(metadata)} objects with tracking")
    index_reid(metadata)
    return jsonify({"message": "Video processed with tracking", "objects": len(metadata)})

@app.route("/search_reid", methods=["GET"])
def search_reid():
    query = request.args.get("query")
    k = int(request.args.get("k", 5))
    results = search_text_reid(query, k)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=False)
