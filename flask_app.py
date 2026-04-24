import html
import logging
import os

import requests
from flask import Flask, session, redirect, render_template, request, url_for, send_from_directory, abort, Response, jsonify
from flask_pymongo import PyMongo

import IR_tools
from utils import get_app_version

APP_VERSION = get_app_version()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "0") == "1"
app.config["SECRET_KEY"] = "safaksdfakjdshfkajshfka" # for session, no actual need for secrecy
DB_SERVER = os.getenv("DB_SERVER", "localhost")
app.config["MONGO_URI"] = f"mongodb://{DB_SERVER}:27017/vatayana"

# TODO: remove once fully done with MongoDB Atlas
# MONGO_CRED = open('mongo_cred/mongo_cred.txt').read().strip()
# app.config["MONGO_URI"] = f"mongodb+srv://tyler:{MONGO_CRED}@sanskrit2.urlv9ek.mongodb.net/vatayana?retryWrites=true&w=majority"


# setup Mongo DB
mongo_db_client = PyMongo(app)
similarity_data = mongo_db_client.db.similarity
print("number of records in collection:", similarity_data.count_documents({}))


# for serving static files from assets folder
@app.route('/assets/<path:name>')
def serve_files(name):
    return send_from_directory('assets', name)

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory('assets', 'robots.txt')

TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "YOUR_SITE_KEY_HERE")
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "YOUR_SECRET_KEY_HERE")

TURNSTILE_EXEMPT = {'/turnstile', '/turnstile/verify', '/robots.txt', '/', '/textView'}

@app.before_request
def block_bots():
    botlist = [
        'Amazonbot', 'Bytespider', 'GPTBot', 'ChatGPT-User', 'CCBot',
        'Google-Extended', 'ClaudeBot', 'anthropic-ai', 'Applebot-Extended',
        'FacebookBot', 'Meta-ExternalAgent', 'PerplexityBot', 'Cohere-ai',
    ]
    user_agent = request.headers.get('User-Agent', '')
    if any(bot in user_agent for bot in botlist):
        abort(403)
    else:
        logger.info("Request handled", extra={'path': request.path, 'method': request.method, 'ip': request.remote_addr})

@app.before_request
def require_turnstile():
    if request.path in TURNSTILE_EXEMPT:
        return
    if request.path.startswith('/assets/'):
        return
    if not session.get('turnstile_passed'):
        session['turnstile_next'] = request.url
        return redirect(url_for('turnstile_challenge'))

# attempt at serving entire folder at once (not yet successful)
# @app.route('/assets/')
# def serve_files():
#     return send_file('assets/index.html')

# this helps app work both remotely and locally
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# variable names for flask.session() object
flask_session_variable_names = [
    "topic_labels",
    "selected_texts",
    "N_tf_idf", "N_sw_w",
    "text_type_toggle"
    ]

def ensure_keys():
    # just in case, make sure all keys in session
    for var_name in flask_session_variable_names:
        if var_name not in session:
            reset_variables()

@app.route('/reset')
def reset_variables():
    session["topic_labels"] = IR_tools.topic_interpretations
    session["selected_texts"] = list(IR_tools.text_abbrev2fn.keys())
    session["N_tf_idf"] = IR_tools.search_N_defaults["N_tf_idf"]
    session["N_sw_w"] = IR_tools.search_N_defaults["N_sw_w"]
    session["text_type_toggle"] = "original"
    session.modified = True
    return redirect(url_for('index'))

@app.route('/about')
def about_page():
    return render_template("about.html", app_version=APP_VERSION)

@app.route('/tutorial')
def tutorial_page():
    return render_template("tutorial.html")

@app.route('/next')
def next_page():
    return render_template("next.html")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/topicVisualizeLDAvis')
def topic_visualize():

    relative_path_to_LDAvis_HTML_fn = "assets/ldavis_prepared_75.html"
    LDAvis_HTML_full_fn = os.path.join(CURRENT_FOLDER, relative_path_to_LDAvis_HTML_fn)
    with open(LDAvis_HTML_full_fn, 'r') as f_in:
        LDAvis_HTML = html.unescape(f_in.read())

    return render_template("topicVisualizeLDAvis.html",
    page_subtitle="topicVisualizeLDAvis",
    LDAvis_HTML=LDAvis_HTML
    )

@app.route('/docExplore', methods=["GET", "POST"])
def doc_explore():

    ensure_keys()

    if request.method == "POST" or 'doc_id' in request.args:
        # NB: not yet supported is sending 'text_abbrv' and 'local_doc_id' via GET

        doc_id = ""
        doc_id_2 = ""

        if 'doc_id' in request.args:
            doc_id = request.args.get("doc_id")
            text_abbreviation = IR_tools.parse_complex_doc_id(doc_id)[0]
            local_doc_id = IR_tools.get_full_local_doc_id(doc_id)
        else:
            text_abbreviation = request.form.get("text_abbreviation_input")
            local_doc_id = request.form.get("local_doc_id_input")
            if local_doc_id not in ['', None]:
                doc_id = text_abbreviation + '_' + local_doc_id

        if 'doc_id_2' in request.args:
            doc_id_2 = request.args.get("doc_id_2")
            local_doc_id_2 = IR_tools.get_full_local_doc_id(doc_id_2)
        else:
            local_doc_id_2 = request.form.get("local_doc_id_input_2")
            if local_doc_id_2 not in ['', None]:
                doc_id_2 = text_abbreviation + '_' + local_doc_id_2

        if 'sw_threshold' in request.args:
            sw_threshold = request.args.get("sw_threshold")
        elif 'sw_threshold' in request.form:
            sw_threshold = request.form.get("sw_threshold")
        else:
            sw_threshold = 50

        if doc_id in ['', None] and local_doc_id in ['', None] and local_doc_id_2 in ['', None]:
            text_id_list = IR_tools.text_doc_ids[text_abbreviation]
            doc_id = text_id_list[0]
            doc_id_2 = text_id_list[-1]

        if 'text_type_toggle' in request.args:
            session["text_type_toggle"] = request.args.get("text_type_toggle")
            session.modified = True
        elif 'text_type_toggle' in request.form:
            session["text_type_toggle"] = request.form.get("text_type_toggle")
            session.modified = True

        def process_doc_ids(doc_id, doc_id_2):

            valid_doc_ids = IR_tools.doc_ids

            if doc_id not in valid_doc_ids:
                return "<br><p>Doc id invalid, please doublecheck.</p>"

            if doc_id_2 == "":
                # single-query mode
                return IR_tools.get_closest_docs(
                    query_id=doc_id,
                    topic_labels=session['topic_labels'],
                    selected_texts=session["selected_texts"],
                    N_tf_idf=session["N_tf_idf"],
                    N_sw_w=session["N_sw_w"],
                    similarity_data=similarity_data,
                    text_type_toggle=session["text_type_toggle"],
                )

            if doc_id_2 not in valid_doc_ids:
                return "<br><p>Second doc id invalid, please doublecheck.</p>"

            text_1 = IR_tools.parse_complex_doc_id(doc_id)[0]
            text_2 = IR_tools.parse_complex_doc_id(doc_id_2)[0]

            if text_1 != text_2:
                return "<br><p>Documents must be from same text.</p>"

            if IR_tools.text_doc_ids[text_1].index(doc_id) >= IR_tools.text_doc_ids[text_1].index(doc_id_2):
                return "<br><p>Please verify sequence of doc id inputs.</p>"

            else:
                # batch-query mode
                best_results = IR_tools.batch_mode(similarity_data, doc_id, doc_id_2, sw_threshold)
                return IR_tools.format_batch_results(best_results, doc_id, doc_id_2, session["selected_texts"])


        docExploreInner_HTML = process_doc_ids(doc_id, doc_id_2)

        return render_template(    "docExplore.html",
                                page_subtitle="docExplore",
                                text_abbreviation=text_abbreviation,
                                local_doc_id=local_doc_id,
                                local_doc_id_2=local_doc_id_2,
                                docExploreInner_HTML=docExploreInner_HTML,
                                sw_threshold=sw_threshold,
                                )

    else: # request.method == "GET" and no arguments or URL query malformed

        return render_template(    "docExplore.html",
                                page_subtitle="docExplore",
                                doc_id="",
                                doc_explore_output="",
                                )

@app.route('/turnstile')
def turnstile_challenge():
    if session.get('turnstile_passed'):
        return redirect(url_for('doc_compare'))
    return render_template('turnstile.html', site_key=TURNSTILE_SITE_KEY)

@app.route('/turnstile/verify', methods=["POST"])
def turnstile_verify():
    token = request.form.get('cf-turnstile-response', '')
    resp = requests.post(
        'https://challenges.cloudflare.com/turnstile/v0/siteverify',
        data={'secret': TURNSTILE_SECRET_KEY, 'response': token},
        timeout=5,
    )
    if resp.json().get('success'):
        session['turnstile_passed'] = True
        next_url = session.pop('turnstile_next', None) or url_for('index')
        return redirect(next_url)
    return redirect(url_for('turnstile_challenge'))

@app.route('/docCompare', methods=["GET", "POST"])
def doc_compare():

    ensure_keys()

    if request.method == "POST" or 'doc_id_1' in request.args:

        if 'doc_id_1' in request.args:
            doc_id_1 = request.args.get("doc_id_1")
            doc_id_2 = request.args.get("doc_id_2")
            text_abbreviation_1 = IR_tools.parse_complex_doc_id(doc_id_1)[0]
            local_doc_id_1 = IR_tools.get_full_local_doc_id(doc_id_1)
            text_abbreviation_2 = IR_tools.parse_complex_doc_id(doc_id_2)[0]
            local_doc_id_2 = IR_tools.get_full_local_doc_id(doc_id_2)
        else:
            text_abbreviation_1 = request.form.get("text_abbreviation_input_1")
            local_doc_id_1 = request.form.get("local_doc_id_input_1")
            doc_id_1 = text_abbreviation_1 + '_' + local_doc_id_1
            text_abbreviation_2 = request.form.get("text_abbreviation_input_2")
            local_doc_id_2 = request.form.get("local_doc_id_input_2")
            doc_id_2 = text_abbreviation_2 + '_' + local_doc_id_2

        valid_doc_ids = IR_tools.doc_ids
        if doc_id_1 == doc_id_2:
            docCompareInner_HTML = "<br><p>Those are the same, please enter two different doc ids to compare.</p>"
        elif doc_id_1 in valid_doc_ids and doc_id_2 in valid_doc_ids:

            docCompareInner_HTML = IR_tools.compare_doc_pair(
                doc_id_1,
                doc_id_2,
                topic_labels=session['topic_labels'],
                selected_texts=session["selected_texts"],
                N_tf_idf=session["N_tf_idf"],
                N_sw_w=session["N_sw_w"],
                similarity_data=similarity_data,
                )
        else:
            docCompareInner_HTML = "<br><p>One or more doc ids invalid, please doublecheck.</p>"

        return render_template(    "docCompare.html",
                                page_subtitle="docCompare",
                                doc_id_1=doc_id_1,
                                doc_id_2=doc_id_2,
                                text_abbreviation_1=text_abbreviation_1,
                                text_abbreviation_2=text_abbreviation_2,
                                local_doc_id_1=local_doc_id_1,
                                local_doc_id_2=local_doc_id_2,
                                docCompareInner_HTML=docCompareInner_HTML,
                                )

    else: # request.method == "GET" or URL query malformed

        return render_template(    "docCompare.html",
                                page_subtitle="docCompare",
                                doc_id_1="",
                                doc_id_2="",
                                text_abbreviation_1="",
                                text_abbreviation_2="",
                                local_doc_id_1="",
                                local_doc_id_2="",
                                doc_explore_output="",
                                )

@app.route('/textView', methods=["GET", "POST"])
def text_view():

    if request.method == "POST" or 'text_abbrv' in request.args or 'doc_id' in request.args:

        if 'doc_id' in request.args:
            doc_id = request.args.get("doc_id")
            text_abbreviation, local_doc_id = IR_tools.parse_complex_doc_id(doc_id)
            return redirect('/textView?text_abbrv={}#{}'.format(text_abbreviation, local_doc_id))
        elif 'text_abbrv' in request.args:
            text_abbreviation =  request.args.get('text_abbrv')
            local_doc_id = ""
        else:
            text_abbreviation = request.form.get("text_abbreviation_input")
            local_doc_id = request.form.get("local_doc_id_input")
            if local_doc_id != "":
                # re-parse to discard unwanted parts of local_doc_id
                doc_id = text_abbreviation + '_' + local_doc_id
                text_abbreviation, local_doc_id = IR_tools.parse_complex_doc_id(doc_id)
            return redirect('/textView?text_abbrv={}#{}'.format(text_abbreviation, local_doc_id))

        text_title = ""
        valid_text_abbrvs = list(IR_tools.text_abbrev2fn.keys())
        disallowed_fulltexts = IR_tools.disallowed_fulltexts
        if text_abbreviation in disallowed_fulltexts:
            text_HTML = "<br><p>sorry, fulltext is not available for these texts at present: " + str(disallowed_fulltexts)[1:-1] + " (see <a href='https://github.com/tylergneill/pramana-nlp/tree/master/data_prep/1_etext_originals' target='_blank'>note</a> for more info)</p>"
        elif text_abbreviation in valid_text_abbrvs:
            text_title = IR_tools.clean_titles[text_abbreviation]
            text_HTML = IR_tools.get_text_view(text_abbreviation)
        else:
            text_HTML = "<br><p>Doc id invalid, please doublecheck.</p>"

        return render_template("textView.html",
                                page_subtitle="textView",
                                text_abbreviation=text_abbreviation,
                                local_doc_id=local_doc_id,
                                text_title=text_title,
                                text_HTML=text_HTML,
                                )

    else: # request.method == "GET" or no URL params

        return render_template(    "textView.html",
                                page_subtitle="textView",
                                text_abbreviation="",
                                local_doc_id="",
                                text_title="",
                                text_HTML="",
                                )

@app.route('/topicAdjust', methods=["GET", "POST"])
def topic_adjust():

    ensure_keys()

    if request.method == "POST":

        topic_label_input = []
        for key, val in request.form.items():
            topic_label_input.append(val)

        session["topic_labels"] = topic_label_input
        session.modified = True

    topicAdjustInner_HTML = IR_tools.format_topic_adjust_output(
        topic_label_input=session["topic_labels"]
        )

    return render_template(    "topicAdjust.html",
                            page_subtitle="topicAdjust",
                            topicAdjustInner_HTML=topicAdjustInner_HTML
                            )


@app.route('/textSelect', methods=["GET", "POST"])
def text_select():

    ensure_keys()

    if request.method == "POST":

        text_select_input = []
        one_text = ""
        all_texts = list(IR_tools.text_abbrev2fn.keys())
        for key, val in request.form.items():
            if key == "select_all_texts":
                # reset to all in default chronological order
                text_select_input = all_texts
        
            elif key == "text_select_checkboxes": # list
                text_select_input = request.form.getlist("text_select_checkboxes")

            elif key == "select_one_text":
                one_text = val
                if one_text in all_texts:
                    text_select_input = [ one_text ]
                else:
                    text_select_input = session["selected_texts"]
                    break
            elif key == "select_earlier_checkbox": # checkbox
                text_select_input = all_texts[:all_texts.index(one_text)] + text_select_input
            elif key == "select_later_checkbox": # checkbox
                text_select_input = text_select_input + all_texts[all_texts.index(one_text)+1:]


        session["selected_texts"] = text_select_input
        session.modified = True

    else:
        pass

    textSelectInner_HTML = IR_tools.format_text_select_output(
        *session["selected_texts"]
        )

    return render_template(    "textSelect.html",
                            page_subtitle="textSelect",
                            textSelectInner_HTML=textSelectInner_HTML,
                            )


@app.route('/searchDepth', methods=["GET", "POST"])
def search_depth():

    ensure_keys()

    if request.method == "POST":

        for key, val in request.form.items():
            if key == "N_tf_idf_slider":
                session["N_tf_idf"] = int(val)
            elif key == "N_sw_w_slider":
                session["N_sw_w"] = int(val)
            elif key == "search_depth_use_defaults":
                session["N_tf_idf"] = IR_tools.search_N_defaults["N_tf_idf"]
                session["N_sw_w"] = IR_tools.search_N_defaults["N_sw_w"]

        session.modified = True

    searchDepthInner_HTML = IR_tools.format_search_depth_output(
        N_tf_idf=session["N_tf_idf"],
        N_sw_w=session["N_sw_w"],
        selected_texts=session["selected_texts"],
        )

    return render_template(    "searchDepth.html",
                            page_subtitle="searchDepth",
                            searchDepthInner_HTML=searchDepthInner_HTML
                            )

@app.route('/api/section_labels.json')
def api_section_labels():
    resp = jsonify(IR_tools.section_labels)
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp

@app.route('/api/abbrv2docs.json')
def api_abbrv2docs():
    resp = jsonify(IR_tools.abbrv2docs)
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp

@app.route('/api/abbrev2title.json')
def api_abbrev2title():
    resp = jsonify(IR_tools.text_abbrev2title)
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp

@app.route('/api/similar_doc_links')
def api_similar_doc_links():
    ensure_keys()
    doc_id = request.args.get('doc_id')
    other_doc_id = request.args.get('other_doc_id')
    if not doc_id or doc_id not in IR_tools.doc_ids:
        return jsonify({'error': 'invalid doc_id'}), 400

    if not similarity_data.find_one({"query_id": doc_id}, {"_id": 1}):
        return jsonify({'found': False})

    similar_doc_links = IR_tools.get_closest_docs(
        query_id=doc_id,
        topic_labels=session['topic_labels'],
        selected_texts=session['selected_texts'],
        N_tf_idf=session['N_tf_idf'],
        N_sw_w=session['N_sw_w'],
        results_as_links_only=True,
        similarity_data=similarity_data,
    )

    if other_doc_id and other_doc_id in similar_doc_links:
        ks = list(similar_doc_links.keys())
        entry = similar_doc_links[other_doc_id]
        rank = ks.index(other_doc_id) + 1
        prev_id = entry['prev']
        next_id = entry['next']
        prev_rank = (ks.index(prev_id) + 1) if prev_id else None
        next_rank = (ks.index(next_id) + 1) if next_id else None
        return jsonify({
            'found': True,
            'rank': rank,
            'total': session['N_sw_w'],
            'prev_doc_id': prev_id,
            'next_doc_id': next_id,
            'prev_rank': prev_rank,
            'next_rank': next_rank,
        })
    else:
        return jsonify({'found': False})


if __name__ == '__main__':
    app.run(debug=True, port=5020)
