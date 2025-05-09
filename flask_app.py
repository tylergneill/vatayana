import logging
import os
import html

from flask import Flask, session, redirect, render_template, request, url_for, send_from_directory, abort
from flask_pymongo import PyMongo

import IR_tools
from utils import get_app_version

APP_VERSION = get_app_version()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app.config["DEBUG"] = True
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

# result = IR_tools.get_closest_docs_with_db(similarity_data, IR_tools.doc_ids[629], priority_texts=['VS'])


# for serving static files from assets folder
@app.route('/assets/<path:name>')
def serve_files(name):
    return send_from_directory('assets', name)

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory('assets', 'robots.txt')

@app.before_request
def block_bots():
    botlist = ['Amazonbot', 'Bytespider']
    user_agent = request.headers.get('User-Agent')
    if any(bot in user_agent for bot in botlist):
        # logger.debug("Bot blocked", extra={'agent': user_agent, 'path': request.path})
        abort(403)
    else:
        logger.info("Request handled", extra={'path': request.path, 'method': request.method, 'ip': request.remote_addr})

# attempt at serving entire folder at once (not yet successful)
# @app.route('/assets/')
# def serve_files():
#     return send_file('assets/index.html')

# this helps app work both publically (e.g. on PythonAnywhere) and locally
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# variable names for flask.session() object
flask_session_variable_names = [
    "doc_id", "doc_id_1", "doc_id_2",
    "text_abbreviation_input", "local_doc_id",
    "topic_labels",
    "priority_texts",
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
    session["doc_id"] = ""; session["doc_id_1"] = ""; session["doc_id_2"] = "",
    session["text_abbreviation_input"] = ""; session["local_doc_id"] = ""
    session["topic_labels"] = IR_tools.topic_interpretations
    session["priority_texts"] = list(IR_tools.text_abbrev2fn.keys())
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

        text_abbreviation_input = ""
        doc_id = ""
        doc_id_2 = ""
        local_doc_id_input = ""
        local_doc_id_input_2 = ""

        if 'doc_id' in request.args:
            doc_id = request.args.get("doc_id")
        else:
            text_abbreviation_input = request.form.get("text_abbreviation_input")
            local_doc_id_input = request.form.get("local_doc_id_input")
            if local_doc_id_input not in ['', None]:
                doc_id = text_abbreviation_input + '_' + local_doc_id_input

        if 'doc_id_2' in request.args:
            doc_id_2 = request.args.get("doc_id_2")
        else:
            local_doc_id_input_2 = request.form.get("local_doc_id_input_2")
            if local_doc_id_input_2 not in ['', None]:
                doc_id_2 = text_abbreviation_input + '_' + local_doc_id_input_2

        if 'sw_threshold' in request.args:
            sw_threshold = request.args.get("sw_threshold")
        elif 'sw_threshold' in request.form:
            sw_threshold = request.form.get("sw_threshold")
        else:
            sw_threshold = 50

        if doc_id in ['', None] and local_doc_id_input in ['', None] and local_doc_id_input_2 in ['', None]:
            text_id_list = IR_tools.text_doc_ids[text_abbreviation_input]
            doc_id = text_id_list[0]
            doc_id_2 = text_id_list[-1]

        if 'text_type_toggle' in request.args:
            session["text_type_toggle"] = request.args.get("text_type_toggle")
            session.modified = True
        elif 'text_type_toggle' in request.form:
            session["text_type_toggle"] = request.form.get("text_type_toggle")
            session.modified = True

        valid_doc_ids = IR_tools.doc_ids
        if (
                doc_id in valid_doc_ids
        ) and (
                doc_id_2 == ""
        ) or (
                (
                    doc_id_2 in valid_doc_ids
                ) and (
                    IR_tools.doc_ids.index(doc_id) < IR_tools.doc_ids.index(doc_id_2)
                )
        ):

            if doc_id_2 != "":

                best_results = IR_tools.batch_mode(similarity_data, doc_id, doc_id_2, sw_threshold)
                docExploreInner_HTML = IR_tools.format_batch_results(best_results, doc_id, doc_id_2, session["priority_texts"])

            else:
                # single-query mode
                docExploreInner_HTML = IR_tools.get_closest_docs(
                    query_id=doc_id,
                    topic_labels=session['topic_labels'],
                    priority_texts=session["priority_texts"],
                    N_tf_idf=session["N_tf_idf"],
                    N_sw_w=session["N_sw_w"],
                    similarity_data=similarity_data,
                    text_type_toggle=session["text_type_toggle"],
                )
        else:
            docExploreInner_HTML = "<br><p>Please verify sequence of two inputs.</p>"
                                   # "Please enter valid doc ids like " + str(IR_tools.ex_doc_ids)[1:-1] + " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> for hints to get started."

        return render_template(    "docExplore.html",
                                page_subtitle="docExplore",
                                text_abbreviation=text_abbreviation_input,
                                local_doc_id=local_doc_id_input,
                                local_doc_id_2=local_doc_id_input_2,
                                docExploreInner_HTML=docExploreInner_HTML,
                                abbrv2docs=IR_tools.abbrv2docs,
                                text_abbrev2title=IR_tools.text_abbrev2title,
                                section_labels=IR_tools.section_labels,
                                )

    else: # request.method == "GET" and no arguments or URL query malformed

        return render_template(    "docExplore.html",
                                page_subtitle="docExplore",
                                doc_id="",
                                doc_explore_output="",
                                abbrv2docs=IR_tools.abbrv2docs,
                                text_abbrev2title=IR_tools.text_abbrev2title,
                                section_labels=IR_tools.section_labels,
                                )

@app.route('/docCompare', methods=["GET", "POST"])
def doc_compare():

    ensure_keys()

    if request.method == "POST" or 'doc_id_1' in request.args:

        if 'doc_id_1' in request.args:
            doc_id_1 = request.args.get("doc_id_1")
            doc_id_2 = request.args.get("doc_id_2")
        else:
            text_abbreviation_input_1 = request.form.get("text_abbreviation_input_1")
            local_doc_id_input_1 = request.form.get("local_doc_id_input_1")
            doc_id_1 = text_abbreviation_input_1 + '_' + local_doc_id_input_1
            text_abbreviation_input_2 = request.form.get("text_abbreviation_input_2")
            local_doc_id_input_2 = request.form.get("local_doc_id_input_2")
            doc_id_2 = text_abbreviation_input_2 + '_' + local_doc_id_input_2

        valid_doc_ids = IR_tools.doc_ids
        sim_btn_left = sim_btn_right = ""
        if doc_id_1 == doc_id_2:
            docCompareInner_HTML = "<br><p>Those are the same, please enter two different doc ids to compare.</p>"
        elif doc_id_1 in valid_doc_ids and doc_id_2 in valid_doc_ids:

            docCompareInner_HTML, sim_btn_left, sim_btn_right = IR_tools.compare_doc_pair(
                doc_id_1,
                doc_id_2,
                topic_labels=session['topic_labels'],
                priority_texts=session["priority_texts"],
                N_tf_idf=session["N_tf_idf"],
                N_sw_w=session["N_sw_w"],
                similarity_data=similarity_data,
                )
        else:
            docCompareInner_HTML = "<br><p>Please enter two valid doc ids like " + str(IR_tools.ex_doc_ids)[1:-1] + " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> for hints to get started.</p>"

        return render_template(    "docCompare.html",
                                page_subtitle="docCompare",
                                doc_id_1=doc_id_1,
                                doc_id_2=doc_id_2,
                                activate_similar_link_buttons_left=sim_btn_left,
                                activate_similar_link_buttons_right=sim_btn_right,
                                docCompareInner_HTML=docCompareInner_HTML,
                                abbrv2docs=IR_tools.abbrv2docs,
                                text_abbrev2title=IR_tools.text_abbrev2title,
                                section_labels=IR_tools.section_labels,
                                )

    else: # request.method == "GET" or URL query malformed

        return render_template(    "docCompare.html",
                                page_subtitle="docCompare",
                                doc_id_1="",
                                doc_id_2="",
                                doc_explore_output="",
                                abbrv2docs=IR_tools.abbrv2docs,
                                text_abbrev2title=IR_tools.text_abbrev2title,
                                section_labels=IR_tools.section_labels,
                                )

@app.route('/textView', methods=["GET", "POST"])
def text_view():

    if request.method == "POST" or 'text_abbrv' in request.args or 'doc_id' in request.args:

        if 'doc_id' in request.args:
            doc_id = request.args.get("doc_id")
            text_abbreviation_input, local_doc_id_input = IR_tools.parse_complex_doc_id(doc_id)
            return redirect('/textView?text_abbrv={}#{}'.format(text_abbreviation_input, local_doc_id_input))
        elif 'text_abbrv' in request.args:
            text_abbreviation_input =  request.args.get('text_abbrv')
            local_doc_id_input = ""
        else:
            text_abbreviation_input = request.form.get("text_abbreviation_input")
            local_doc_id_input = request.form.get("local_doc_id_input")
            if local_doc_id_input != "":
                # re-parse to discard unwanted parts of local_doc_id_input
                doc_id = text_abbreviation_input + '_' + local_doc_id_input
                text_abbreviation_input, local_doc_id_input = IR_tools.parse_complex_doc_id(doc_id)
            return redirect('/textView?text_abbrv={}#{}'.format(text_abbreviation_input, local_doc_id_input))

        text_title = ""
        valid_text_abbrvs = list(IR_tools.text_abbrev2fn.keys())
        disallowed_fulltexts = IR_tools.disallowed_fulltexts
        if text_abbreviation_input in disallowed_fulltexts:
            text_HTML = "<br><p>sorry, fulltext is not available for these texts at present: " + str(disallowed_fulltexts)[1:-1] + " (see <a href='https://github.com/tylergneill/pramana-nlp/tree/master/data_prep/1_etext_originals' target='_blank'>note</a> for more info)</p>"
        elif text_abbreviation_input in valid_text_abbrvs:
            text_title = IR_tools.clean_titles[text_abbreviation_input]
            text_HTML = IR_tools.get_text_view(text_abbreviation_input)
        else:
            text_HTML = "<br><p>Please enter valid doc ids like " + str(IR_tools.ex_doc_ids)[1:-1] + " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> for hints to get started.</p>"

        return render_template("textView.html",
                                page_subtitle="textView",
                                text_abbreviation=text_abbreviation_input,
                                local_doc_id=local_doc_id_input,
                                text_title=text_title,
                                text_HTML=text_HTML,
                                abbrv2docs=IR_tools.abbrv2docs,
                                text_abbrev2title=IR_tools.text_abbrev2title,
                                section_labels=IR_tools.section_labels,
                                )

    else: # request.method == "GET" or no URL params

        return render_template(    "textView.html",
                                page_subtitle="textView",
                                text_abbreviation="",
                                local_doc_id="",
                                text_title="",
                                text_HTML="",
                                abbrv2docs=IR_tools.abbrv2docs,
                                text_abbrev2title=IR_tools.text_abbrev2title,
                                section_labels=IR_tools.section_labels,
                                )

@app.route('/BrucheionAlign')
def Brucheion_align():

    relative_path_to_assets = "assets"
    full_path_to_assets = os.path.join(CURRENT_FOLDER, relative_path_to_assets)

    relative_path_to_Brucheion_HTML_body_fn = "assets/Brucheion.html"
    Brucheion_HTML_body_full_fn = os.path.join(CURRENT_FOLDER, relative_path_to_Brucheion_HTML_body_fn)
    with open(Brucheion_HTML_body_full_fn, 'r') as f_in:
        Brucheion_HTML_body = html.unescape(f_in.read())

    return render_template(    "BrucheionAlign.html",
                            assets_path=relative_path_to_assets,
                            # page_subtitle="alignFancy",
                            Brucheion_HTML_body=Brucheion_HTML_body
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


@app.route('/textPrioritize', methods=["GET", "POST"])
def text_prioritize():

    ensure_keys()

    if request.method == "POST":

        priority_texts_input = []
        one_text = ""
        all_texts = list(IR_tools.text_abbrev2fn.keys())
        for key, val in request.form.items():
            if key == "prioritize_all_texts":
                # reset to all in default chronological order
                priority_texts_input = all_texts
            # elif key == "prioritize_none":
            #     # string for current text prepared in form
            #     priority_texts_input = [ val ]

            elif key == "priority_checkboxes": # list
                priority_texts_input = request.form.getlist("priority_checkboxes")

            elif key == "prioritize_one_text":
                one_text = val
                if one_text in all_texts:
                    priority_texts_input = [ one_text ]
                else:
                    priority_texts_input = session["priority_texts"]
                    break
            elif key == "prioritize_earlier_checkbox": # checkbox
                priority_texts_input = all_texts[:all_texts.index(one_text)] + priority_texts_input
            elif key == "prioritize_later_checkbox": # checkbox
                priority_texts_input = priority_texts_input + all_texts[all_texts.index(one_text)+1:]


        session["priority_texts"] = priority_texts_input
        session.modified = True

    else:
        pass

    textPrioritizeInner_HTML = IR_tools.format_text_prioritize_output(
        *session["priority_texts"]
        )

    return render_template(    "textPrioritize.html",
                            page_subtitle="textPrioritize",
                            textPrioritizeInner_HTML=textPrioritizeInner_HTML
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
        priority_texts=session["priority_texts"],
        )

    return render_template(    "searchDepth.html",
                            page_subtitle="searchDepth",
                            searchDepthInner_HTML=searchDepthInner_HTML
                            )

if __name__ == '__main__':
    app.run(debug=True, port=5020)
