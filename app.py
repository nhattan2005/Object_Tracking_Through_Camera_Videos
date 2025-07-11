from flask import Flask, request, jsonify
from detect import extract_objects
from indexer import index_images
from search import search_text
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

@app.route("/process_video", methods=["POST"])
def process_video():
    video_path = request.json.get("video_path")
    metadata = extract_objects(video_path)
    index_images(metadata)
    return jsonify({"message": "Video processed", "objects": len(metadata)})

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    k = int(request.args.get("k", 5))

    print(f"[SEARCH] Query: {query}")
    try:
        results = search_text(query, k)
        print(f"[SEARCH] Found {len(results)} results")
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()  # <--- thêm dòng này để hiện lỗi chi tiết
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
