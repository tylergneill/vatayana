import os
import html

from flask import Flask, session, redirect, render_template, request, url_for, send_from_directory
from flask_pymongo import PyMongo

import IR_tools

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "safaksdfakjdshfkajshfka"  # for session, no actual need for secrecy
MONGO_CRED = open('mongo_cred.txt').read().strip()
# app.config["MONGO_URI"] = "mongodb://localhost:27017/my_db"
app.config["MONGO_URI"] = f"mongodb+srv://tyler:{MONGO_CRED}@sanskrit2.urlv9ek.mongodb.net/vatayana?retryWrites=true&w=majority"


# setup Mongo DB
mongo_db_client = PyMongo(app)
similarity_data = mongo_db_client.db.my_collection  # local
# similarity_data = mongo_db_client.db.similarity  # remote
print("num of records in collection:", similarity_data.count_documents({}))


# for serving static files from assets folder
@app.route('/assets/<path:name>')
def serve_files(name):
    return send_from_directory('assets', name)


# this helps app work both publicly (e.g. on PythonAnywhere) and locally
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# variable names for flask.session() object
flask_session_variable_names = [
    "doc_id", "doc_id_1", "doc_id_2",
    "text_abbreviation_input", "local_doc_id",
    "topic_labels",
    "priority_texts",
    "n_tf_idf_shallow", "n_sw_w_shallow",
    "n_tf_idf_deep", "n_sw_w_deep",
    "search_depth_default",
]


def ensure_keys():
    # just in case, make sure all keys in session
    for var_name in flask_session_variable_names:
        if var_name not in session:
            reset_variables()


@app.route('/reset')
def reset_variables():
    session["doc_id"] = ""
    session["doc_id_1"] = ""
    session["doc_id_2"] = "",
    session["text_abbreviation_input"] = ""
    session["local_doc_id"] = ""
    session["topic_labels"] = IR_tools.topic_interpretations
    session["priority_texts"] = list(IR_tools.text_abbrev2fn.keys())
    session["n_tf_idf_shallow"] = IR_tools.search_n_defaults["n_tf_idf_shallow"]
    session["n_tf_idf_deep"] = IR_tools.search_n_defaults["n_tf_idf_deep"]
    session["n_sw_w_shallow"] = IR_tools.search_n_defaults["n_sw_w_shallow"]
    session["n_sw_w_deep"] = IR_tools.search_n_defaults["n_sw_w_deep"]
    session["search_depth_default"] = "shallow"
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
    return render_template("index.html", page_subtitle="🪟")


@app.route('/topicVisualizeLDAvis')
def topic_visualize():
    relative_path_to_ldavis_html_fn = "assets/ldavis_prepared_75.html"
    ldavis_html_full_fn = os.path.join(CURRENT_FOLDER, relative_path_to_ldavis_html_fn)
    with open(ldavis_html_full_fn, 'r') as f_in:
        ldavis_html = html.unescape(f_in.read())
    return render_template(
        "topicVisualizeLDAvis.html",
        page_subtitle="topicVisualizeLDAvis",
        ldavis_html=ldavis_html,
    )


@app.route('/docExplore', methods=["GET", "POST"])
def doc_explore():

    ensure_keys()

    if request.method == "POST" or 'doc_id' in request.args:
        # NB: not yet supported is sending text_abbrv and local_doc_id via GET

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
                # batch mode

                best_results = IR_tools.batch_mode(similarity_data, doc_id, doc_id_2, sw_threshold)
                doc_explore_inner_html = IR_tools.format_batch_results(
                    best_results,
                    doc_id,
                    doc_id_2,
                    session["priority_texts"],
                )

            else:
                # single-query mode

                doc_explore_inner_html = IR_tools.get_closest_docs(
                    query_id=doc_id,
                    topic_labels=session['topic_labels'],
                    priority_texts=session["priority_texts"],
                    n_tf_idf=session["n_tf_idf_"+session["search_depth_default"]],
                    n_sw_w=session["n_sw_w_"+session["search_depth_default"]],
                    similarity_data=similarity_data,
                )

        else:

            doc_explore_inner_html = "<br><p>Please verify sequence of two inputs.</p>"
            # doc_explore_inner_html = f"<br><p>Please enter valid doc ids like " \
            #                          f"{str(IR_tools.ex_doc_ids)[1:-1]} etc.</p>" \
            #                          f"<p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> and " \
            #                          f"<a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> " \
            #                          f"for hints to get started."

        return render_template(
            "docExplore.html",
            page_subtitle="docExplore",
            text_abbreviation=text_abbreviation_input,
            local_doc_id=local_doc_id_input,
            local_doc_id_2=local_doc_id_input_2,
            doc_explore_inner_html=doc_explore_inner_html,
            abbrv2docs=IR_tools.abbrv2docs,
            text_abbrev2title=IR_tools.text_abbrev2title,
            section_labels=IR_tools.section_labels,
        )

    else:
        # request.method == "GET" and no arguments or URL query malformed

        return render_template(
            "docExplore.html",
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

            doc_compare_inner_html = "<br><p>Those are the same, please enter two different doc ids to compare.</p>"

        elif doc_id_1 in valid_doc_ids and doc_id_2 in valid_doc_ids:

            doc_compare_inner_html, sim_btn_left, sim_btn_right = IR_tools.compare_doc_pair(
                doc_id_1,
                doc_id_2,
                topic_labels=session['topic_labels'],
                priority_texts=session["priority_texts"],
                n_tf_idf=session["n_tf_idf_"+session["search_depth_default"]],
                n_sw_w=session["n_sw_w_"+session["search_depth_default"]],
                similarity_data=similarity_data,
            )

        else:

            doc_compare_inner_html = f"<br><p>Please enter two valid doc ids like " \
                                     f"{str(IR_tools.ex_doc_ids)[1:-1]} etc.</p>" \
                                     f"<p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> " \
                                     f"and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> " \
                                     f"for hints to get started.</p>"

        return render_template(
            "docCompare.html",
            page_subtitle="docCompare",
            doc_id_1=doc_id_1,
            doc_id_2=doc_id_2,
            activate_similar_link_buttons_left=sim_btn_left,
            activate_similar_link_buttons_right=sim_btn_right,
            doc_compare_inner_html=doc_compare_inner_html,
            abbrv2docs=IR_tools.abbrv2docs,
            text_abbrev2title=IR_tools.text_abbrev2title,
            section_labels=IR_tools.section_labels,
        )

    else:
        # request.method == "GET" or URL query malformed

        return render_template(
            "docCompare.html",
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

            text_abbreviation_input = request.args.get('text_abbrv')
            local_doc_id_input = ""

        else:

            text_abbreviation_input = request.form.get("text_abbreviation_input")
            local_doc_id_input = request.form.get("local_doc_id_input")
            if local_doc_id_input != "":
                # reparse to discard unwanted parts of local_doc_id_input
                doc_id = text_abbreviation_input + '_' + local_doc_id_input
                text_abbreviation_input, local_doc_id_input = IR_tools.parse_complex_doc_id(doc_id)
            return redirect('/textView?text_abbrv={}#{}'.format(text_abbreviation_input, local_doc_id_input))

        text_title = ""
        valid_text_abbrvs = list(IR_tools.text_abbrev2fn.keys())
        disallowed_fulltexts = IR_tools.disallowed_fulltexts

        if text_abbreviation_input in disallowed_fulltexts:

            text_html = "<br><p>sorry, fulltext is not available for these texts at present: "
            text_html += str(disallowed_fulltexts)[1:-1]
            disallowed_list = "https://github.com/tylergneill/pramana-nlp/tree/master/data_prep/1_etext_originals"
            text_html += f" (see <a href='{disallowed_list}' target='_blank'>note</a> for more info)</p>"

        elif text_abbreviation_input in valid_text_abbrvs:

            text_title = IR_tools.text_abbrev2fn[text_abbreviation_input]
            text_html = IR_tools.get_text_view(text_abbreviation_input)

        else:

            text_html = "<br><p>Please enter valid doc ids like "
            text_html += str(IR_tools.ex_doc_ids)[1:-1]
            text_html += " etc.</p><p>See <a href='assets/doc_id_list.txt' target='_blank'>doc id list</a> " \
                         "and <a href='assets/corpus_texts.txt' target='_blank'>corpus text list</a> " \
                         "for hints to get started.</p>"

        return render_template(
            "textView.html",
            page_subtitle="textView",
            text_abbreviation=text_abbreviation_input,
            local_doc_id=local_doc_id_input,
            text_title=text_title,
            text_html=text_html,
            abbrv2docs=IR_tools.abbrv2docs,
            text_abbrev2title=IR_tools.text_abbrev2title,
            section_labels=IR_tools.section_labels,
        )

    else:
        # request.method == "GET" or no URL params

        return render_template(
            "textView.html",
            page_subtitle="textView",
            text_abbreviation="",
            local_doc_id="",
            text_title="",
            text_html="",
            abbrv2docs=IR_tools.abbrv2docs,
            text_abbrev2title=IR_tools.text_abbrev2title,
            section_labels=IR_tools.section_labels,
        )


@app.route('/BrucheionAlign')
def brucheion_align():

    relative_path_to_assets = "assets"

    relative_path_to_brucheion_html_body_fn = "assets/Brucheion.html"
    brucheion_html_body_full_fn = os.path.join(CURRENT_FOLDER, relative_path_to_brucheion_html_body_fn)
    with open(brucheion_html_body_full_fn, 'r') as f_in:
        brucheion_html_body = html.unescape(f_in.read())

    return render_template(
        "BrucheionAlign.html",
        assets_path=relative_path_to_assets,
        # page_subtitle="alignFancy",
        brucheion_html_body=brucheion_html_body,
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

    topic_adjust_inner_html = IR_tools.format_topic_adjust_output(
        topic_label_input=session["topic_labels"],
    )

    return render_template(
        "topicAdjust.html",
        page_subtitle="topicAdjust",
        topic_adjust_inner_html=topic_adjust_inner_html,
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

            elif key == "priority_checkboxes":  # list

                priority_texts_input = request.form.getlist("priority_checkboxes")

            elif key == "prioritize_one_text":

                one_text = val

                if one_text in all_texts:

                    priority_texts_input = [one_text]

                else:

                    priority_texts_input = session["priority_texts"]
                    break

            elif key == "prioritize_earlier_checkbox":  # checkbox

                priority_texts_input = all_texts[:all_texts.index(one_text)] + priority_texts_input

            elif key == "prioritize_later_checkbox":  # checkbox

                priority_texts_input = priority_texts_input + all_texts[all_texts.index(one_text)+1:]

        session["priority_texts"] = priority_texts_input
        session.modified = True

    text_prioritize_inner_html = IR_tools.format_text_prioritize_output(
        *session["priority_texts"],
    )

    return render_template(
        "textPrioritize.html",
        page_subtitle="textPrioritize",
        text_prioritize_inner_html=text_prioritize_inner_html
    )


@app.route('/searchDepth', methods=["GET", "POST"])
def search_depth():

    ensure_keys()

    if request.method == "POST":

        for key, val in request.form.items():

            if key == "n_tf_idf_shallow_slider":
                session["n_tf_idf_shallow"] = int(val)
            elif key == "n_tf_idf_deep_slider":
                session["n_tf_idf_deep"] = int(val)
            elif key == "n_sw_w_shallow_slider":
                session["n_sw_w_shallow"] = int(val)
            elif key == "n_sw_w_deep_slider":
                session["n_sw_w_deep"] = int(val)
            elif key == "search_depth_use_defaults":
                session["n_tf_idf_shallow"] = IR_tools.search_n_defaults["n_tf_idf_shallow"]
                session["n_tf_idf_deep"] = IR_tools.search_n_defaults["n_tf_idf_deep"]
                session["n_sw_w_shallow"] = IR_tools.search_n_defaults["n_sw_w_shallow"]
                session["n_sw_w_deep"] = IR_tools.search_n_defaults["n_sw_w_deep"]
            elif key == "search_depth_radio":
                session["search_depth_default"] = val

        session.modified = True

    search_depth_inner_html = IR_tools.format_search_depth_output(
        n_tf_idf_shallow=session["n_tf_idf_shallow"],
        n_sw_w_shallow=session["n_sw_w_shallow"],
        n_tf_idf_deep=session["n_tf_idf_deep"],
        n_sw_w_deep=session["n_sw_w_deep"],
        priority_texts=session["priority_texts"],
        search_depth_default=session["search_depth_default"],
    )

    return render_template(
        "searchDepth.html",
        page_subtitle="searchDepth",
        search_depth_inner_html=search_depth_inner_html,
    )
