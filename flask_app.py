import os
import html
import re

from datetime import datetime, date
from flask import Flask, redirect, render_template, request, url_for, session, send_from_directory, send_file, make_response

import IR_tools

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "safaksdfhjlakjdshfkajshfka" # for session, no actual need for secrecy

# for serving static files from assets folder
@app.route('/assets/<path:name>')
def serve_files(name):
    return send_from_directory('assets', name)

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
    "topic_weights",
    "topic_labels",
    "priority_texts",
    "topic_toggle_value"
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
    session["topic_weights"] = IR_tools.topic_weights_default.tolist()
    session["topic_labels"] = IR_tools.topic_interpretations
    session["priority_texts"] = list(IR_tools.text_abbrev2fn.keys())
    session["topic_toggle_value"] = True
    session.modified = True
    return redirect(url_for('index'))

@app.route('/about')
def about_page():
    return render_template("about.html")

@app.route('/tutorial')
def tutorial_page():
    return render_template("tutorial.html")

@app.route('/next')
def next_page():
    return render_template("next.html")

@app.route('/')
def index():
    return render_template("index.html",
    page_subtitle="🪟"
    )

@app.route('/topicVisualizeLDAvis')
def vatayana_topic_visualize():

    relative_path_to_LDAvis_HTML_fn = "assets/ldavis_prepared_50.html"
    LDAvis_HTML_full_fn = os.path.join(CURRENT_FOLDER, relative_path_to_LDAvis_HTML_fn)
    with open(LDAvis_HTML_full_fn, 'r') as f_in:
        LDAvis_HTML = html.unescape(f_in.read())

    return render_template("topicVisualizeLDAvis.html",
    page_subtitle="topicVisualizeLDAvis",
    LDAvis_HTML=LDAvis_HTML
    )

@app.route('/docExplore', methods=["GET", "POST"])
def vatayana_doc_explore():

    ensure_keys()

    if request.method == "POST" or 'doc_id' in request.args:

        if 'doc_id' in request.form:
            doc_id = request.form.get("doc_id")
        elif 'doc_id' in request.args:
            doc_id = request.args.get("doc_id")

        valid_doc_ids = IR_tools.doc_ids
        if doc_id in valid_doc_ids:

            auto_reweight_topics_option = False
            if auto_reweight_topics_option:
                topic_weights = IR_tools.auto_reweight_topics(doc_id)
                session['topic_weights'] = topic_weights
                session.modified = True

            docExploreInner_HTML = IR_tools.get_closest_docs(
                doc_id,
                topic_weights=session['topic_weights'],
                topic_labels=session['topic_labels'],
                priority_texts=session["priority_texts"],
                topic_toggle_value=session["topic_toggle_value"]
                )
        else:
            docExploreInner_HTML = "<br><p>Please enter valid doc ids like " + str(IR_tools.ex_doc_ids)[1:-1] + " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> for hints to get started.</p>"

        return render_template(    "docExplore.html",
                                page_subtitle="docExplore",
                                doc_id=doc_id,
                                docExploreInner_HTML=docExploreInner_HTML
                                )

    else: # request.method == "GET" or URL query malformed

        return render_template(    "docExplore.html",
                                page_subtitle="docExplore",
                                doc_id="",
                                doc_explore_output=""
                                )

@app.route('/docCompare', methods=["GET", "POST"])
def vatayana_doc_compare():

    ensure_keys()

    if request.method == "POST" or 'doc_id_1' in request.args:

        doc_id_1 = doc_id_2 = ""
        if 'doc_id_1' in request.form:
            doc_id_1 = request.form.get("doc_id_1")
            doc_id_2 = request.form.get("doc_id_2")
        elif 'doc_id_1' in request.args:
            doc_id_1 = request.args.get("doc_id_1")
            doc_id_2 = request.args.get("doc_id_2")

        valid_doc_ids = IR_tools.doc_ids
        sim_btn_left = sim_btn_right = ""
        if doc_id_1 == doc_id_2:
            output_HTML = "<br><p>Those are the same, please enter two different doc ids to compare.</p>"
        elif doc_id_1 in valid_doc_ids and doc_id_2 in valid_doc_ids:
            # output_HTML = "<br><p>Good, those are valid.</p>"

            auto_reweight_topics_option = False
            if auto_reweight_topics_option:
                topic_weights = IR_tools.auto_reweight_topics(doc_id_1)
                session['topic_weights'] = topic_weights
                session.modified = True

            docCompareInner_HTML, sim_btn_left, sim_btn_right = IR_tools.compare_doc_pair(
                doc_id_1,
                doc_id_2,
                topic_weights=session['topic_weights'],
                topic_labels=session['topic_labels'],
                priority_texts=session["priority_texts"],
                topic_toggle_value=session["topic_toggle_value"]
                )
        else:
            docCompareInner_HTML = "<br><p>Please enter two valid doc ids like " + str(IR_tools.ex_doc_ids)[1:-1] + " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> for hints to get started.</p>"

        return render_template(    "docCompare.html",
                                page_subtitle="docCompare",
                                doc_id_1=doc_id_1,
                                doc_id_2=doc_id_2,
                                activate_similar_link_buttons_left=sim_btn_left,
                                activate_similar_link_buttons_right=sim_btn_right,
                                docCompareInner_HTML=docCompareInner_HTML
                                )

    else: # request.method == "GET" or URL query malformed

        return render_template(    "docCompare.html",
                                page_subtitle="docCompare",
                                doc_id_1="",
                                doc_id_2="",
                                doc_explore_output=""
                                )

@app.route('/textView', methods=["GET", "POST"])
def vatayana_text_view():

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
                import pdb; pdb.set_trace()
                doc_id = text_abbreviation_input + '_' + local_doc_id_input
                text_abbreviation_input, local_doc_id_input = IR_tools.parse_complex_doc_id(doc_id)
            return redirect('/textView?text_abbrv={}#{}'.format(text_abbreviation_input, local_doc_id_input))

        text_title = ""
        valid_text_abbrvs = list(IR_tools.text_abbrev2fn.keys())
        disallowed_fulltexts = IR_tools.disallowed_fulltexts
        if text_abbreviation_input in disallowed_fulltexts:
            text_HTML = "<br><p>sorry, fulltext is not available for these texts at present: " + str(disallowed_fulltexts)[1:-1] + " (see <a href='https://github.com/tylergneill/pramana-nlp/tree/master/data_prep/1_etext_originals' target='_blank'>note</a> for more info)</p>"
        elif text_abbreviation_input in valid_text_abbrvs:
            text_title = IR_tools.text_abbrev2fn[text_abbreviation_input]
            text_HTML = IR_tools.get_text_view(text_abbreviation_input)
        else:
            text_HTML = "<br><p>Please enter valid doc ids like " + str(IR_tools.ex_doc_ids)[1:-1] + " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> for hints to get started.</p>"

        return render_template("textView.html",
                                page_subtitle="textView",
                                text_abbreviation=text_abbreviation_input,
                                local_doc_id=local_doc_id_input,
                                text_title=text_title,
                                text_HTML=text_HTML
                                )

    else: # request.method == "GET" or no URL params

        return render_template(    "textView.html",
                                page_subtitle="textView",
                                text_abbreviation="",
                                local_doc_id="",
                                text_title="",
                                text_HTML=""
                                )

@app.route('/BrucheionAlign')
def vatayana_Brucheion_align():

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
def vatayana_topic_adjust():

    ensure_keys()

    if request.method == "POST":

        topic_weight_input = []
        topic_label_input = []
        for key, val in request.form.items():
            if key == "topic_wt_slider_all":
                topic_weight_input = list(IR_tools.new_full_vector( IR_tools.K, float(val) ))
                topic_label_input = session["topic_labels"]
            elif key.startswith("topic_wt_slider_"):
                topic_weight_input.append(float(val)) # not sure why 1s come back as int
            elif key.startswith("topic_label_"):
                topic_label_input.append(val)

        session["topic_weights"] = topic_weight_input
        session["topic_labels"] = topic_label_input
        session.modified = True

    else:
        pass

    topicAdjustInner_HTML = IR_tools.format_topic_adjust_output(
        topic_weight_input=session["topic_weights"],
        topic_label_input=session["topic_labels"]
        )

    return render_template(    "topicAdjust.html",
                            page_subtitle="topicAdjust",
                            topicAdjustInner_HTML=topicAdjustInner_HTML
                            )


@app.route('/textPrioritize', methods=["GET", "POST"])
def vatayana_text_prioritize():

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

@app.route('/topicToggle', methods=["GET", "POST"])
def vatayana_topic_toggle():

    ensure_keys()

    if request.method == "POST":

        if "topic_toggle_checkbox" in request.form:
            topic_toggle_value = True
        else:
            topic_toggle_value = False

        session["topic_toggle_value"] = topic_toggle_value
        session.modified = True

    topicToggleInner_HTML = IR_tools.format_topic_toggle_output(
        session["topic_toggle_value"]
        )

    return render_template(    "topicToggle.html",
                            page_subtitle="topicToggle",
                            topicToggleInner_HTML=topicToggleInner_HTML
                            )
