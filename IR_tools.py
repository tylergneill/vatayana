import os
import json
import re
import math
import string
from typing import List, Dict, Optional, Union, Tuple, Any

import numpy as np

from collections import OrderedDict, defaultdict
from fastdist import fastdist
from string import Template
from datetime import datetime, date
from collatex import *
from lxml import etree
from difflib import SequenceMatcher
from flask_pymongo.wrappers import Collection as PymongoCollection


########################################################
# set up absolute path, JSON loader, and HTML templates
########################################################

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def load_dict_from_json(relative_path_fn):
    json_full_fn = os.path.join(CURRENT_FOLDER, relative_path_fn)
    with open(json_full_fn,'r') as f_in:
        loaded_dict = json.loads( f_in.read() )
    return loaded_dict

HTML_templates = {}
template_names = [
    'docExploreInner',
    'docExploreBatchInner',
    'docCompareInner',
    'topicAdjustInner',
    'textPrioritizeInner',
    'searchDepthInner'
    ]
for template_name in template_names:
    HTML_template_relative_path = 'templates/{}.html'.format(template_name)
    HTML_template_fn = os.path.join(CURRENT_FOLDER, HTML_template_relative_path)
    with open(HTML_template_fn,'r') as f_in:
        HTML_templates[template_name] = Template(f_in.read())


######################################
# load accessory pre-processsing data
######################################

# load lookup table of section headers by doc_id
section_labels = load_dict_from_json("assets/section_labels.json")
# e.g. section_labels[DOC_ID] = STRING

# load sister dict of doc_fulltext with original punctuation (only some!) and unsplit text
doc_original_fulltext = load_dict_from_json("assets/doc_original_fulltext.json")
# e.g. doc_original_fulltext[DOC_ID] = STRING


######################################
# load corpus and topic modeling data
######################################

def process_theta_data() -> Tuple[Any]:
    theta_fn = 'assets/theta.tsv'
    theta_fn_full_path = os.path.join(CURRENT_FOLDER, theta_fn)
    with open(theta_fn_full_path,'r') as f_in:
        theta_rows = f_in.read().split('\n')
    theta_rows.pop(-1); # blank final row
    theta_rows.pop(0); # unwanted header row with topic abbreviations (store same from phi data)
    theta_rows.pop(0); # unwanted second header row with "!ctsdata" and alpha values

    # store theta data (doc ids, doc full-text, and theta numbers)
    doc_ids = []
    doc_fulltext = OrderedDict() # e.g. doc_fulltext[DOC_ID]
    thetas = {} # e.g. theta[DOC_ID]
    for row in theta_rows:
        cells = row.split('\t') # must have been converted to TSV first!
        doc_id, doc_text, theta_values = cells[1], cells[2], cells[3:]
        # don't need cells[0] which would be doc_num
        K = len(cells) - 3
        doc_ids.append(doc_id)
        doc_fulltext[doc_id] = doc_text.replace('*','') # HACK, should be cleaned in data itself
        thetas[doc_id] = [ float(th) for th in theta_values ]

    corpus_vocab = set()
    for text in doc_fulltext.values():
        corpus_vocab.update(text.split())

    # count each term's document frequency
    doc_freq = {}  # e.g. doc_freq[WORD] = INT for each word in vocab
    for doc_id in doc_ids:
        doc_text = doc_fulltext[doc_id]
        doc_tokens = doc_text.split()
        doc_unique_words = list(set(doc_tokens))
        for word in doc_unique_words:
            # increment doc_freq tally
            if word in doc_freq:
                doc_freq[word] += 1
            else:
                doc_freq[word] = 1
    num_docs = len(doc_ids)

    # calculate inverse document frequencies
    IDF = {}  # e.g. IDF[WORD] = FLOAT for each word in vocab
    for word in corpus_vocab:
        IDF[word] = math.log(num_docs / doc_freq[word])

    # prepare list of stopwords (and temporarily also other error-words to exclude)
    stopwords = ['iti', 'na', 'ca', 'api', 'eva', 'tad', 'tvāt', 'tat', 'hi', 'ādi', 'tu',
                 'vā']  # used in topic modeling
    # NB: stopwords are those entirely excluded from topic modeling, such that they have no associated phi numbers
    error_words = [':', '*tat', 'eva*', '*atha', ')']  # should fix in the data!

    # prepare corpus_vocab_reduced to use for high-dimensional document vectors

    too_common_doc_freq_cutoff = 0.27  # smaller cutoff is more exclusive

    too_rare_doc_freq_cutoff = 0.00300  # larger cutoff is more exclusive

    # e.g., for 20k-doc corpus with vocab 79,606, keeping constant too_common_doc_freq_cutoff
    # 0.01000 (   721,  0.91%) >>  0.5-sec wait
    # 0.00300 ( 2,175,  2.73%) >>  1.5-sec wait
    # 0.00150 ( 3,931,  4.94%) >>  2.7-sec wait
    # 0.00030 (12,967, 16.29%) >> 12.0-sec wait

    corpus_vocab_reduced = [
        word
        for word in corpus_vocab
        if not (word in stopwords + error_words
                or doc_freq[word] / num_docs < too_rare_doc_freq_cutoff
                or doc_freq[word] / num_docs > too_common_doc_freq_cutoff)
    ]

    return (doc_ids, doc_fulltext, thetas, IDF, corpus_vocab_reduced, stopwords)

doc_ids, doc_fulltext, thetas, IDF, corpus_vocab_reduced, stopwords = process_theta_data()
num_docs = len(doc_ids)

# to be calibrated for PythonAnywhere
topic_secs_per_comparison   = 0.000007 #    7 microseconds
tf_idf_secs_per_comparison  = 0.000315 #  315 microseconds
sw_w_secs_per_comparison    = 0.004513 # 4513 microseconds

search_N_defaults = {
    "N_tf_idf" : 500,
    "N_sw_w" : 25,
}


ex_doc_ids = ['NBhū_104,6^1', 'SŚP_2.21', 'MV_1,i_5,i^1']

disallowed_fulltexts = ['PVin','HB','PSṬ','NV']

# save fresh doc_id list to file
doc_id_list_relative_path_fn = 'assets/doc_id_list.txt'
doc_id_list_full_fn = os.path.join(CURRENT_FOLDER, doc_id_list_relative_path_fn)
with open(doc_id_list_full_fn,'w') as f_out:
    f_out.write('\n'.join(doc_ids))


def parse_complex_doc_id(doc_id):
# NB: returns only first original doc id from any resizing modifications
    if doc_id is None:
        return None, None
    first_underscore_pos = doc_id.find('_')
    work_abbrv = doc_id[:first_underscore_pos]
    local_doc_id = re.search('[^_\^:]+', doc_id[first_underscore_pos+1:]).group()
    return work_abbrv, local_doc_id


def get_full_local_doc_id(doc_id):
    work_abbrv, _ = parse_complex_doc_id(doc_id)
    if work_abbrv is None:
        return None
    return doc_id[len(work_abbrv)+1:]


def build_similarity_doc_nav(elem_list):
    nav = {}
    for i, doc_id in enumerate(elem_list):
        nav[doc_id] = {
            'prev': elem_list[i - 1] if i > 0 else None,
            'next': elem_list[i + 1] if i < len(elem_list) - 1 else None,
        }
    return nav


def build_by_work_doc_nav(elem_list):
    """
    Given a flat list of doc_ids like ['NBhū_1,1', 'NBhū_1,2', ... , 'PVin_I,034,i', ...],
    returns a dict mapping each doc_id to a dict of
      {
        'first':  <first in that work>,
        'prev':   <prev in that work> or None,
        'next':   <next in that work> or None,
        'last':   <last  in that work>
      }
    """
    # 1) group by work-prefix
    groups = defaultdict(list)
    for doc_id in elem_list:
        work, _ = parse_complex_doc_id(doc_id)
        groups[work].append(doc_id)

    # 2) build navigation for each group
    nav = {}
    for work, ids in groups.items():
        first, last = ids[0], ids[-1]
        for i, doc_id in enumerate(ids):
            prev_id = ids[i-1] if i > 0           else None
            next_id = ids[i+1] if i < len(ids)-1  else None
            nav[doc_id] = {
                'first': first,
                'prev':  prev_id,
                'next':  next_id,
                'last':  last
            }

    return nav

doc_links = build_by_work_doc_nav(doc_ids)

def clean_title(raw_title):
    cleaned_title = raw_title.replace('sāṃkhya_', '')
    cleaned_title = cleaned_title.replace('_', "’s ")
    cleaned_title = string.capwords(cleaned_title)
    return cleaned_title

# load lookup table of filenames by conventional text abbreviation
text_abbrev2fn = load_dict_from_json("assets/text_abbreviations_IASTreduced.json") # for accessing files
text_abbrev2title = load_dict_from_json("assets/text_abbreviations.json") # for human eyes
clean_titles = {k: clean_title(v) for k, v in text_abbrev2title.items()}
# e.g. text_abbrev2fn[TEXT_ABBRV] = STRING
# don't sort these yet because they're in chronological order for presenting prioritization options

# create lookup table of local_doc_ids by text abbreviation
abbrv2docs = defaultdict(lambda:[])
for doc_id in doc_ids:
    first_underscore = doc_id.find('_')
    abbrv, local_doc_id = doc_id[:first_underscore], doc_id[first_underscore+1:]
    abbrv2docs[abbrv].append(local_doc_id)

# save fresh corpus text list to file
corpus_texts_list_relative_path_fn = 'assets/corpus_texts.txt'
corpus_texts_list_full_fn = os.path.join(CURRENT_FOLDER, corpus_texts_list_relative_path_fn)
with open(corpus_texts_list_full_fn,'w') as f_out:
    f_out.write('\n'.join([abbrv+'\t'+fn for (abbrv, fn) in text_abbrev2title.items()]))


###################################################
# load post-processed illustrations of topic meaning
###################################################

topic_top_words_fn = 'assets/topic_top_words.txt'
topic_top_words_fn_full_path = os.path.join(CURRENT_FOLDER, topic_top_words_fn)
with open(topic_top_words_fn_full_path,'r') as f_in:
    topic_top_words = f_in.readlines()

topic_interpretation_fn = 'assets/topic_interpretations_28k_75t.txt'
topic_interpretation_fn_full_path = os.path.join(CURRENT_FOLDER, topic_interpretation_fn)
with open(topic_interpretation_fn_full_path,'r') as f_in:
    topic_interpretations = f_in.readlines()

topic_wordclouds_relative_path = 'assets/topic_wordclouds'
topic_wordclouds_full_path = os.path.join(CURRENT_FOLDER, topic_wordclouds_relative_path)
topic_wordcloud_fns = [ os.path.join(topic_wordclouds_relative_path, img_fn) # relative for img src
                            for img_fn in os.listdir(topic_wordclouds_full_path) # full to reach img_fn here
                            if os.path.isfile(os.path.join(topic_wordclouds_full_path, img_fn))
                            and img_fn != '.DS_Store'
                        ]
topic_wordcloud_fns.sort()


#######################################################
# create/load simple HTML transformations for textView
#######################################################

stored_textView_HTML_relative_path = "assets/textView_HTML"
stored_textView_HTML_full_path = os.path.join(CURRENT_FOLDER, stored_textView_HTML_relative_path)

stored_textView_HTML_fns = [ os.path.join(stored_textView_HTML_full_path, HTML_fn)
                            for HTML_fn in os.listdir(stored_textView_HTML_full_path)
                            if os.path.isfile(os.path.join(stored_textView_HTML_full_path, HTML_fn))
                            and HTML_fn != '.DS_Store'
                            ]

def format_text_view(text_abbreviation):

    # use text_abbreviation to read in text string
    text_fn = text_abbrev2fn[text_abbreviation] + '.txt'
    relative_path_to_text = "assets/texts"
    text_full_fn = os.path.join(CURRENT_FOLDER, relative_path_to_text, text_fn)
    with open(text_full_fn,'r') as f_in:
        text_string = f_in.read()

    # adjust for <...>
    # otherwise, e.g. something that begins <s ...> or <S ...> (e.g. 'Seite') will be interpreted as strikethrough
    def escape_angle_brackets(text):
        # Replace < with &lt;
        escaped_text = text.replace('<', '&lt;')
        # Replace > with &gt;
        escaped_text = escaped_text.replace('>', '&gt;')
        return escaped_text
    escaped_text_string = escape_angle_brackets(text_string)

    # wrap in single div
    text_HTML = f"<div class='content'>{escaped_text_string}</div>"

    # wrap {...} content in <h2>
    text_HTML = re.sub("{([^}]*?)}", "<h2 id='\\1'>\\1</h2>", text_HTML)

    # wrap [...] in <h3> and also use content for id attribute and link
    h2s = re.findall("\[([^\]]*?)\]", text_HTML)
    work_doc_ids = [    doc_id
                        for doc_id in doc_ids
                        if parse_complex_doc_id(doc_id)[0] == text_abbreviation
                        ]
    for h2 in h2s:
        links_addendum = "<small>"
        relevant_work_doc_ids = [    doc_id for doc_id in work_doc_ids
                                    if parse_complex_doc_id(doc_id)[1] == h2
                                ]
        links_addendum += '  '.join( [ "(<a href='docExplore?doc_id={}'>{}</a>)".format(doc_id, doc_id) for doc_id in relevant_work_doc_ids ] )
        links_addendum += "</small>"
        try:
            text_HTML = re.sub("\[({})\]".format(h2), "<h3 id='\\1'>\\1 {}</h3>".format(links_addendum), text_HTML)
        except:
            # this detects encoding errors in the original text which mess up the HTML formatting
            import pdb; pdb.set_trace()

    # wrap content in <p>
    pattern_replacements = [
        (r"^([^\n\t<]+)$", r"<p>\1</p>"),
        (r"^(\t[^\n<]+)$", r"<p class='l'>\1</p>"),
        (r"(</h3>\t)(.+)$", r"\1<p class='l'>\2</p>"),
    ]
    for pattern, replacement in pattern_replacements:
        text_HTML = re.sub(pattern, replacement, text_HTML, flags=re.MULTILINE)

    return text_HTML

def get_text_view(text_abbreviation):

    viewText_HTML_fn = text_abbrev2fn[text_abbreviation] + '.html'
    viewText_HTML_fn_full_path = os.path.join(stored_textView_HTML_full_path, viewText_HTML_fn)
    if viewText_HTML_fn_full_path in stored_textView_HTML_fns:
        # print("FOUND STORED HTML")
        with open(viewText_HTML_fn_full_path, 'r') as f_in:
            viewText_HTML = f_in.read()
            return viewText_HTML
    else:
        # print("MAKING NEW HTML")
        viewText_HTML = format_text_view(text_abbreviation)
        with open(viewText_HTML_fn_full_path, 'w') as f_out:
            f_out.write(viewText_HTML)
        return viewText_HTML

# for offline use
# create more textView HTML pages
# import pdb; pdb.set_trace()
# for txt_abbrv in tqdm(list(text_abbrev2fn.keys())):
#     if txt_abbrv not in disallowed_fulltexts:
#         get_text_view(txt_abbrv)



##########################################################
# various functions for interface modes (docExplore etc.)
##########################################################

# handy general function for getting indices of N max (descending) values in list
def indices_of_top_N_elements(L, N):
    return sorted(range(len(L)), key=lambda x: L[x], reverse=True)[:N]

# characterize document by (pythonic index!) numbers (INT) for max_N topics over threshold and corresponding percentage (STRING)
# for topic plot caption
def get_top_topic_indices(doc_id, max_N=5, threshold=0.03):
# return list of tuples of type (%d, %s)
    indices_of_dominant_N_topics = indices_of_top_N_elements(L=thetas[doc_id], N=max_N)
    qualifying_indices = [  i
                            for i in indices_of_dominant_N_topics
                            if thetas[doc_id][i] >= threshold
                            ]
    return qualifying_indices


def sort_score_dict(dictionary: Dict[str, Union[float, List[Union[float, str]]]]) -> Dict:
    """
    Takes a dict of the form {'key_1': value_1, 'key_n': value_n} and returns sorted according to values.
    Also works if dict values are a list of the form [float, str] rather than just a float (as for sw scores).
    """
    return dict( sorted(dictionary.items(), key=lambda item: item[1], reverse=True) )

def rank_all_candidates_by_topic_similarity(query_id):

    if doc_fulltext[query_id] == '': return {}

    query_vector = np.array(thetas[query_id])
    topic_similiarity_score = {} # e.g. topic_similiarity_score[DOC_ID] = FLOAT
    for doc_id in doc_ids:
        candidate_vector = np.array(thetas[doc_id])
        # use doc_fulltext to check if empty bc exact empty theta vector depends on alpha type (asymmetric etc.)
        if doc_fulltext[doc_id] == '':
            topic_similiarity_score[doc_id] = 0
        else:
            topic_similiarity_score[doc_id] = round(1-fastdist.cosine(query_vector, candidate_vector), 4)

    topic_similiarity_score.pop(query_id) # remove query itself

    # return sorted dict in descending order by value
    sorted_results = sort_score_dict(topic_similiarity_score)
    return sorted_results


def divide_doc_id_list_by_work_priority(list_of_doc_ids_to_prune, priority_works):
    prioritized = []
    secondary = []
    for doc_id in list_of_doc_ids_to_prune:
        if parse_complex_doc_id(doc_id)[0] in priority_works:
            prioritized.append(doc_id)
        else:
            secondary.append(doc_id)
    return prioritized, secondary


def get_TF_IDF_vector(doc_id):
    # calculates tf.idf vector for any given doc
    # returns as numpy array

    doc_text = doc_fulltext[doc_id]
    doc_words = doc_text.split()
    unique_doc_words = list(set(doc_words))

    total_doc_word_len = len(doc_words)

    TF_IDF_dict = {} # e.g. TF_IDF_dict[WORD] = [FLOAT, FLOAT, FLOAT, FLOAT, ... FLOAT]
    for word in unique_doc_words:
        TF_d_w = doc_words.count(word) / total_doc_word_len
        TF_IDF_dict[word] = TF_d_w * IDF[word]

    TF_IDF_vector = np.zeros( len(corpus_vocab_reduced) )
    # e.g. TF_IDF_vector[WORD] = [0, 0, 0, ... FLOAT, 0, 0, ... FLOAT, 0, ... 0]

    for word in TF_IDF_dict.keys():
        if word in corpus_vocab_reduced:
            i = corpus_vocab_reduced.index(word) # alphabetical index
            TF_IDF_vector[i] = TF_IDF_dict[word]

    return TF_IDF_vector


# for offline use:
text_doc_ids = {}
text_abbrevs = list(text_abbrev2fn.keys())
for text_abbrev in text_abbrevs:
    text_doc_ids[text_abbrev] = [
        doc_id for doc_id in doc_ids if parse_complex_doc_id(doc_id)[0] == text_abbrev
    ]


def get_tiny_TF_IDF_vectors(doc_id_1, doc_id_2):
    # returns numpy arrays

    doc_text_1 = doc_fulltext[doc_id_1]
    doc_text_2 = doc_fulltext[doc_id_2]

    doc_text_1_words = doc_text_1.split()
    doc_text_2_words = doc_text_2.split()

    mini_vocab = list(set(doc_text_1_words + doc_text_2_words))
    mini_vocab.sort()

    TF_IDF_vector_1 = np.zeros( len(mini_vocab) )
    TF_IDF_vector_2 = np.zeros( len(mini_vocab) )

    for i, word in enumerate(mini_vocab):

        TF_1 = doc_text_1_words.count(word) / len(doc_text_1_words)
        TF_2 = doc_text_2_words.count(word) / len(doc_text_2_words)

        TF_IDF_vector_1[i] = TF_1 * IDF[word]
        TF_IDF_vector_2[i] = TF_2 * IDF[word]

    return TF_IDF_vector_1, TF_IDF_vector_2

def rank_candidates_by_tiny_TF_IDF_similarity(query_id, candidate_ids):

    if doc_fulltext[query_id] == '' or candidate_ids == []: return {}

    TF_IDF_comparison_scores = {} # e.g. TF_IDF_comparison_scores[DOC_ID] = FLOAT
    for doc_id in candidate_ids:
        query_vector, candidate_vector = get_tiny_TF_IDF_vectors(query_id, doc_id)
        TF_IDF_comparison_scores[doc_id] = round(1-fastdist.cosine(query_vector, candidate_vector), 4)

    sorted_results = sort_score_dict(TF_IDF_comparison_scores)
    return sorted_results


def group_doc_ids_by_work(*doc_ids_to_do):
    doc_ids_grouped_by_work = {}
    for doc_id in doc_ids_to_do:
        work_name = parse_complex_doc_id(doc_id)[0]
        if work_name not in doc_ids_grouped_by_work.keys():
            doc_ids_grouped_by_work[work_name] = []
        doc_ids_grouped_by_work[work_name].append(doc_id)
    return doc_ids_grouped_by_work


# HTML formatting functions

def format_topic_explore_links(topic_index):
    return """<a
href='topicVisualizeLDAvis#topic={}&lambda=0.8&term=' target='_blank'>#{:02}</a> <a
href='{}' title='{}' target='_wordcloud'>☁️</a>""".format(
        # topic_interpretations[topic_index],
        topic_index+1,
        topic_index+1,
        topic_wordcloud_fns[topic_index],
        topic_top_words[topic_index]
        )

def format_top_topic_summary(doc_id, top_topic_indices, topic_labels):
    top_topic_summary_HTML = "<style type='text/css'>td {padding:0 15px;}</style>"
    top_topic_summary_HTML += "<div class='container' style='margin-left: 40px;'><table>"
    top_topic_summary_HTML += ''.join(
        [ """<tr>
                <td><h3><small>{:.1%}</small></h3></td>
                <td><h3><small>{}</small></h3></td>
                <td><h3><small>({})</small></h3></td>
            </tr>""".format(
                thetas[doc_id][i],
                topic_labels[i],
                format_topic_explore_links(i)
                )
        for i in top_topic_indices
        ] )
    top_topic_summary_HTML += "</table></div>"
    return top_topic_summary_HTML

def format_docView_link(doc_id):
    # looks like doc_id
    clean_work_name = clean_titles[parse_complex_doc_id(doc_id)[0]]
    return "<a href='docExplore?doc_id=%s' title='%s {%s}'>%s</a>" % (doc_id, clean_work_name, section_labels[doc_id], doc_id)

def format_textView_link(doc_id):
    # each one looks like fixed string "txtVw"
    work_abbrv, local_doc_id = parse_complex_doc_id(doc_id)
    return "<a href='textView?text_abbrv=%s#%s' target='textView%s'>txtVw</a>" % (work_abbrv, local_doc_id, work_abbrv)

def format_docCompare_link(doc_id_1, doc_id_2, display_string="dcCp", title=""):
    # each one looks like fixed string "dcCp" unless otherwise specified
    return "<a href='docCompare?doc_id_1=%s&doc_id_2=%s' target='docCompare' title='%s'>%s</a>" % (doc_id_1, doc_id_2, title, display_string)

def format_similarity_result_columns(query_id, priority_results_list_content, secondary_results_list_content):

    # priority
    table_header_row =     """<thead>
                                <tr>
                                    <th>rank</th>
                                    <th>similar doc</th>
                                    <th>topic score</th>
                                    <th>word score</th>
                                    <th>phrase score</th>
                                    <th>phrase preview</th>
                                </tr>
                            </thead>"""
    table_row_template =     """<tr>
                                    <td>%d</td>
                                    <td>%s</td>
                                    <td>%.2f</td>
                                    <td>%s</td>
                                    <td>%s</td>
                                    <td>%s</td>
                                </tr>"""
    #                                     <th>links</th>
    #                                     <td>%s&nbsp;&nbsp;%s</td>

    priority_col_HTML = "<table id='priority_col_table' class='display'>"
    priority_col_HTML += table_header_row + "<tbody>"

    priority_col_HTML += ''.join( [
        table_row_template % (
            i+1,
            format_docView_link(doc_id),
            results[0], # topic score
            results[1], # tf-idf score
            results[2][0], # alignment score
            format_docCompare_link(query_id, doc_id, display_string=results[2][1][:25], title=results[2][1]),  # alignment phrase, max 25 chars
            # format_textView_link(doc_id),
            # format_docCompare_link(query_id, doc_id)
            )
        for i, (doc_id, results) in enumerate(priority_results_list_content.items())
        ] )
    priority_col_HTML += "</tbody></table>"

    # secondary

    secondary_col_HTML = "<table id='secondary_col_table' class='display'>"
    secondary_col_HTML += table_header_row + "<tbody>"

    secondary_col_HTML += ''.join( [
        table_row_template % (
            i+1,
            format_docView_link(doc_id),
            result, # topic only
            "", # no tf-idf
            "", # no alignment
            format_textView_link(doc_id),
            format_docCompare_link(query_id, doc_id)
            )
        for i, (doc_id, result) in enumerate(secondary_results_list_content.items())
        ] )
    secondary_col_HTML += "</tbody></table>"

    return priority_col_HTML, secondary_col_HTML


def rank_candidates_by_sw_w_alignment_score(query_id, candidate_ids, sw_w_score_threshold=30):

    if doc_fulltext[query_id] == '' or candidate_ids == []: return {}

    sw_alignment_scores = {}
    for i, doc_id in enumerate(candidate_ids):

        text_1, text_2 = doc_fulltext[query_id], doc_fulltext[doc_id]
        subseq1_pos, subseq2_pos, subseq1_len, subseq2_len, score = sw_align(text_1, text_2, words=True)
        if (subseq1_pos, subseq2_pos, subseq1_len, subseq2_len, score) == (0, 0, 0, 0, 0):
            sw_alignment_scores[doc_id] = [0.0, ""]
        else:
            subseq1 = ' '.join( text_1.split(' ')[subseq1_pos:subseq1_pos+subseq1_len] )
            subseq2 = ' '.join( text_2.split(' ')[subseq2_pos:subseq2_pos+subseq2_len] )
            subseq1_pos, subseq2_pos, subseq1_len, subseq2_len, raw_score = sw_align(subseq1, subseq2, words=False)
            sw_w_score = raw_score / 10
            if sw_w_score >= sw_w_score_threshold:
                sw_alignment_scores[doc_id] = [sw_w_score, subseq1]
            else:
                sw_alignment_scores[doc_id] = [sw_w_score, ""]

    sorted_results = sort_score_dict(sw_alignment_scores)
    return sorted_results

def calc_dur(start, end):
    delta = datetime.combine(date.today(), end) - datetime.combine(date.today(), start)
    duration_secs = delta.seconds + delta.microseconds / 1000000
    return duration_secs


def truncate_dict(dictionary: Dict, n: int) -> Dict:
    """
    Returns the first n items of the dict. For use with sorted dicts.
    """
    return {
        k: v
        for (k, v) in list(dictionary.items())[:n]
    }

N_TDIDF_SAVE_LIMIT = 2500
N_SW_SAVE_LIMIT = 500
def get_closest_docs_with_db(
        similarity_data: PymongoCollection,
        query_id,
        N_tfidf=N_TDIDF_SAVE_LIMIT,
        N_sw=N_SW_SAVE_LIMIT,
        priority_texts: List[str]=list(text_abbrev2fn.keys()),
    ) -> Dict[str, Dict[str, float]]:
    # start = datetime.now().time()
    if not (
            record := similarity_data.find_one({"query_id": query_id})
    # ) or not (
    #         len(topic_similar_docs := record["similar_docs"]["topic"]) != len(doc_ids)
    ):
        # simply do from scratch
        similar_docs = calculate_similar_docs(query_id, N_tfidf, N_sw)

    else:
        topic_similar_docs = rank_all_candidates_by_topic_similarity(query_id)

        # all topic comparisons done

        tf_idf_similar_docs = record["similar_docs"]["tf_idf"]
        sw_w_similar_docs = record["similar_docs"]["sw_w"]
        additional_tfidf = None
        if not (len(tf_idf_similar_docs) >= N_tfidf):
            # not enough tf-idf comparisons already done, do more
            additional_tfidf = rank_candidates_by_tiny_TF_IDF_similarity(
                query_id,
                list(topic_similar_docs.keys())[len(tf_idf_similar_docs):N_tfidf]
            )
            # print("len(additional_tfidf):", len(additional_tfidf))
            tf_idf_similar_docs = dict(tf_idf_similar_docs, **additional_tfidf)  # can't use .update()
            tf_idf_similar_docs = sort_score_dict(tf_idf_similar_docs)

            # cache for sw_w now unreliable since new possibilities just added
            # existing scores still correct, just not necessarily correct rank
            # immediately refresh sw_w cache by replacing with new scores as needed, keep at same size
            existing_sw_cache_size = len(sw_w_similar_docs)
            doc_ids_for_sw_comparison = [
                doc_id for doc_id in truncate_dict(tf_idf_similar_docs, existing_sw_cache_size)
                if doc_id not in sw_w_similar_docs
            ]
            additional_sw = rank_candidates_by_sw_w_alignment_score(
                query_id,
                doc_ids_for_sw_comparison,
            )
            # print("len(additional_sw) due to additional_tfidf:", len(additional_sw))
            sw_w_similar_docs = dict(sw_w_similar_docs, **additional_sw)  # can't use .update()
            sw_w_similar_docs = sort_score_dict(sw_w_similar_docs)
            sw_w_similar_docs = truncate_dict(sw_w_similar_docs, existing_sw_cache_size)

            # enough tf-idf comparisons done now

        if not (len(sw_w_similar_docs) >= N_sw):
            # not enough sw comparisons already done, do more

            additional_sw = rank_candidates_by_sw_w_alignment_score(
                query_id,
                list(tf_idf_similar_docs.keys())[len(sw_w_similar_docs):N_sw]
            )
            # print("len(additional_sw):", len(additional_sw))
            sw_w_similar_docs = dict(sw_w_similar_docs, **additional_sw)  # can't use .update()
            sw_w_similar_docs = sort_score_dict(sw_w_similar_docs)

        # enough sw comparisons done now

        similar_docs = {
            'topic': topic_similar_docs,
            'tf_idf': tf_idf_similar_docs,
            'sw_w': sw_w_similar_docs,
        }

    # truncate what gets saved to prevent writing too much to db
    similar_docs_to_save = {
        # 'topic': similar_docs['topic'],  # dropped because too big
        'tf_idf': truncate_dict(similar_docs['tf_idf'], N_TDIDF_SAVE_LIMIT),
        'sw_w': truncate_dict(similar_docs['sw_w'], N_SW_SAVE_LIMIT),
    }

    # save results
    query = {"query_id": query_id}
    update = {"$set": {"similar_docs": similar_docs_to_save}}
    insertion_result = similarity_data.update_one(
        query,
        update,
        upsert=True
    )

    return similar_docs


def calculate_similar_docs(query_id, N_tfidf=4300, N_sw=200) -> Dict[str, Dict[str, float]]:
    topic_similar_docs = rank_all_candidates_by_topic_similarity(query_id)
    tf_idf_similar_docs = rank_candidates_by_tiny_TF_IDF_similarity(
        query_id,
        list(topic_similar_docs.keys())[:N_tfidf]
    )
    sw_w_similar_docs = rank_candidates_by_sw_w_alignment_score(
        query_id,
        list(tf_idf_similar_docs.keys())[:N_sw]
    )
    similar_docs = {
        'topic': topic_similar_docs,
        'tf_idf': tf_idf_similar_docs,
        'sw_w': sw_w_similar_docs
    }
    return similar_docs


def get_closest_docs(   query_id,
                        topic_labels=topic_interpretations,
                        priority_texts=list(text_abbrev2fn.keys()),
                        N_tf_idf=search_N_defaults["N_tf_idf"],
                        N_sw_w=search_N_defaults["N_sw_w"],
                        results_as_links_only=False,
                        similarity_data: Optional[PymongoCollection]=None,
                        batch_mode: Optional[bool]=False,
                        sw_w_min_threshold: Optional[int]=50,
                        text_type_toggle="original",
                        ):

    non_priority_texts = [text for text in list(text_abbrev2fn.keys()) if text not in priority_texts]

    start0 = datetime.now().time()
    # get num of docs in priority_texts to use for computation time calculations
    num_priority_docs = sum([ num_docs_by_text[text_name] for text_name in priority_texts ])

    # start1 = datetime.now().time()
    # end1 = datetime.now().time()
    # topic_time = calc_dur(start1, end1)

    # handle blank
    if doc_fulltext[query_id] == '':
        results_HTML = HTML_templates['docExploreInner'].substitute(
            query_id = query_id,
            query_work_name=(query_work_name := parse_complex_doc_id(query_id)[0]),
            query_id_local=(query_id_local := get_full_local_doc_id(query_id)),
            text_display_name=clean_titles[query_work_name],
            first_doc_id=get_full_local_doc_id(doc_links[query_id]['first']),
            prev_doc_id=get_full_local_doc_id(doc_links[query_id]['prev']),
            next_doc_id=get_full_local_doc_id(doc_links[query_id]['next']),
            last_doc_id=get_full_local_doc_id(doc_links[query_id]['last']),
            query_text_pos=abbrv2docs[query_work_name].index(query_id_local) + 1,
            query_text_doc_count=len(abbrv2docs[query_work_name]),
            query_section=section_labels[query_id],
            query_original_fulltext = doc_original_fulltext[query_id],
            query_segmented_fulltext = '',
            top_topics_summary='',
            priority_results_list_content = '',
            secondary_results_list_content = '',
            priority_texts=str(priority_texts),
            non_priority_texts=str(non_priority_texts),
            text_type_toggle=text_type_toggle,
            )
        return results_HTML

    # use get_closest_docs_with_db
    if similarity_data != None:

        similar_docs: Dict[str, Dict[str, float]] = get_closest_docs_with_db(
            similarity_data,
            query_id,
            N_tfidf = N_tf_idf,
            N_sw = N_sw_w,
            priority_texts=priority_texts, # not used!
            )

        priority_topic_candidates = similar_docs['topic']
        tf_idf_candidates = similar_docs['tf_idf']
        sw_w_alignment_candidates = similar_docs['sw_w']

        # do NOT prioritize by text at all

    else:

        # prioritize by text and by topic similarity

        # get N preliminary candidates by topic score (dimensionality = K, fast)

        start1 = datetime.now().time()

        all_topic_candidates = rank_all_candidates_by_topic_similarity(
            query_id
            )

        end1 = datetime.now().time()
        topic_time = calc_dur(start1, end1)

        # prioritize candidates by text name
        priority_candidate_ids, secondary_candidate_ids = divide_doc_id_list_by_work_priority(
            list(all_topic_candidates.keys()),
            priority_texts
            )
        priority_topic_candidates = { doc_id: all_topic_candidates[doc_id]
            for doc_id in priority_candidate_ids
            }
        secondary_topic_candidates = { doc_id: all_topic_candidates[doc_id]
            for doc_id in secondary_candidate_ids
            }

        # limit further computation to only top N_tf_idf of sorted candidates (minus query itself)
        pruned_priority_topic_candidates = truncate_dict(priority_topic_candidates, N_tf_idf)

        start2 = datetime.now().time()

        # further rank candidates by tiny tf-idf
        tf_idf_candidates = rank_candidates_by_tiny_TF_IDF_similarity(
            query_id,
            list(pruned_priority_topic_candidates.keys())
            )

        end2 = datetime.now().time()
        tf_idf_time = calc_dur(start2, end2)

        # would like to bottom of priority list other priority-text docs for which only topics compared
        # but very inefficient on page render
        # for now, thereofre, shunt these to secondary results (end of list for now)...
        for k, v in priority_topic_candidates.items():
            if k not in tf_idf_candidates:
                secondary_topic_candidates[k] = v

            # limit further computation to only top N_sw_w of sorted candidates
        pruned_tf_idf_candidates = truncate_dict(tf_idf_candidates, N_sw_w)

        start3 = datetime.now().time()

        # further rank candidates by sw_w
        sw_w_alignment_candidates = rank_candidates_by_sw_w_alignment_score(
            query_id,
            list(pruned_tf_idf_candidates.keys())
        )

        end3 = datetime.now().time()
        sw_time = calc_dur(start3, end3)

        end0 = datetime.now().time()
        overall_time = calc_dur(start0, end0)
        print(f"topic_time: {topic_time} sec, len(all_topic_candidates): {len(all_topic_candidates)},  {topic_time/len(all_topic_candidates)} s / topic comparison")
        print(f"tf_idf_time: {tf_idf_time} sec, len(tf_idf_candidates): {len(tf_idf_candidates)}, {tf_idf_time/len(tf_idf_candidates)} s / tf_idf comparison")
        print(f"sw_time: {sw_time} sec, len(pruned_tf_idf_candidates): {len(pruned_tf_idf_candidates)}, {sw_time/len(pruned_tf_idf_candidates)} s / sw comparison")
        print(f"overall_time: {overall_time} sec")

    # post-processing

    # for those that have sw score but no tf-idf due to truncation limits, do one-off tf-idf, resort
    for k in sw_w_alignment_candidates:
        if k not in tf_idf_candidates:
            doc_1_TF_IDF_vector, doc_2_TF_IDF_vector = get_tiny_TF_IDF_vectors(query_id, k)
            tf_idf_candidates[k] = round(1 - fastdist.cosine(doc_1_TF_IDF_vector, doc_2_TF_IDF_vector), 4)
    tf_idf_candidates = sort_score_dict(tf_idf_candidates)

    # post-ranking, convert to strings (round to two decimal places, empty replaces 0.0)
    for k,v in tf_idf_candidates.items():
        if v == 0.0: tf_idf_candidates[k] = ""
        else: tf_idf_candidates[k] = "{:.2f}".format(tf_idf_candidates[k])

    # post-ranking, convert numbers to strings (empty replaces 0.0, no need for rounding)
    for k,score_phrase_pair in sw_w_alignment_candidates.items():
        if score_phrase_pair[0] == 0.0: sw_w_alignment_candidates[k] = ("", "")
        else: sw_w_alignment_candidates[k] = (str(sw_w_alignment_candidates[k][0]), sw_w_alignment_candidates[k][1])

    # again add blank entries to bottom of list for all docs for which sw_w comparison not performed
    for k in tf_idf_candidates.keys(): # contains priority_topic_candidates.keys() too
        if k not in sw_w_alignment_candidates:
            sw_w_alignment_candidates[k] = ("", "")

    # thus final results list has sw_w candidates on top, tf_idf candidates after that, and priority_topic_candidates after that

    priority_ranked_results_ids = list(sw_w_alignment_candidates.keys())

    priority_ranked_results_complete = {
        k: (priority_topic_candidates[k], tf_idf_candidates[k], sw_w_alignment_candidates[k])
        for k in priority_ranked_results_ids
    }

    if similarity_data != None:
        # need to filter for priority texts at this point
        # this is relatively computationally expensive!

        start4 = datetime.now().time()

        FILTRATION_LIMIT = 2000
        # truncate results at reasonable limit to speed up following steps
        # filter out non-priority texts
        priority_ranked_results_complete = {
            k: v for k, v in list(priority_ranked_results_complete.items())[:FILTRATION_LIMIT]
            if parse_complex_doc_id(k)[0] in priority_texts
        }

        LOADING_LIMIT = 500
        # further truncate what gets loaded on page
        priority_ranked_results_complete = truncate_dict(priority_ranked_results_complete, LOADING_LIMIT)
        # TODO: add "Load more" button that loads rest into table (repurpose "secondary" structure)
        # additional_ranked_results_complete = {
        #     k: v for k, v in list(priority_ranked_results_complete.items())[LOADING_LIMIT:]
        # }

        end4 = datetime.now().time()
        filtering_time = calc_dur(start4, end4)
        # print("filtering_time:", filtering_time)

    if results_as_links_only:
        similarity_result_doc_links = build_similarity_doc_nav(list(priority_ranked_results_complete.keys())[:N_sw_w])
        return similarity_result_doc_links

    if batch_mode:
        # pick out absolute best results
        best_results = {}
        for doc_id_2, result in priority_ranked_results_complete.items():
            if float(result[2]) >= sw_w_min_threshold:
                best_results[doc_id_2] = result
            else:
                break

        # return query's best results as simple HTML rows
        results_HTML = ""
        for doc_id_2, result in best_results.items():
            results_HTML += """
            <tr align="center">
              <td>{}</td>
              <td>{}</td>
              <td>{}</td>
              <td>{}</td>
              <td>{}</td>
            </tr>
            """.format(
                    query_id,
                    doc_id_2,
                    result[0],
                    result[1],
                    result[2],
                    # subseq_in_text_1,
                    # link,
                    # <td align="left">{}</td>
                    # <td>{}</td>
            )

    else:

        priority_col_HTML, secondary_col_HTML = format_similarity_result_columns(
            query_id,
            priority_ranked_results_complete,
            # secondary_topic_candidates
            {}
        )
        if priority_col_HTML == "": priority_col_HTML = "<p>(none)</p>"
        # if secondary_col_HTML == "": secondary_col_HTML = "<p>(none)</p>"
        secondary_col_HTML = "<p>(none)</p>" # just neutralize for now until i can make faster
        results_HTML = HTML_templates['docExploreInner'].substitute(
                            query_id = query_id,
                            query_work_name=(query_work_name := parse_complex_doc_id(query_id)[0]),
                            query_id_local=(query_id_local := get_full_local_doc_id(query_id)),
                            text_display_name=clean_titles[query_work_name],
                            first_doc_id=get_full_local_doc_id(doc_links[query_id]['first']),
                            prev_doc_id=get_full_local_doc_id(doc_links[query_id]['prev']),
                            next_doc_id=get_full_local_doc_id(doc_links[query_id]['next']),
                            last_doc_id=get_full_local_doc_id(doc_links[query_id]['last']),
                            query_text_pos=abbrv2docs[query_work_name].index(query_id_local) + 1,
                            query_text_doc_count=len(abbrv2docs[query_work_name]),
                            query_section = section_labels[query_id],
                            query_original_fulltext = doc_original_fulltext[query_id],
                            query_segmented_fulltext = doc_fulltext[query_id],
                            top_topics_summary=format_top_topic_summary(
                                query_id,
                                get_top_topic_indices(query_id, max_N=5, threshold=0.03),
                                topic_labels=topic_labels
                                ),
                            priority_col_content = priority_col_HTML,
                            secondary_col_content = secondary_col_HTML,
                            priority_texts=str(priority_texts),
                            non_priority_texts=str(non_priority_texts),
                            text_type_toggle=text_type_toggle,
                            )

    # import pdb; pdb.set_trace()

    return results_HTML


def batch_mode(
        similarity_data,
        query_doc_id_start,
        query_doc_id_end,
        sw_score_threshold,
) -> List[Dict[str, Union[str, float]]]:
    query_doc_id_range = range(doc_ids.index(query_doc_id_start), doc_ids.index(query_doc_id_end) + 1)
    query_doc_ids = [doc_ids[i] for i in query_doc_id_range]

    start0 = datetime.now().time()
    query = {"query_id": {"$in": query_doc_ids}}
    projection = {"_id": 0, "query_id": 1, "similar_docs.tf_idf": 1, "similar_docs.sw_w": 1}
    all_records = similarity_data.find(query, projection)
    # print("fetch records:", calc_dur(start0, datetime.now().time()))

    start1 = datetime.now().time()
    records_dict = {
        record['query_id']: {
            'tf_idf': record['similar_docs']['tf_idf'],
            'sw_w': record['similar_docs']['sw_w'],
        } for record in list(all_records)
    }
    # print("dict records:", calc_dur(start1, datetime.now().time()))

    start2 = datetime.now().time()
    ks = sorted(list(records_dict.keys()))
    sorted_records_dict = {k: records_dict[k] for k in ks}
    # print("sort records:", calc_dur(start2, datetime.now().time()))

    start3 = datetime.now().time()
    best_results: List[Dict[str, Union[str, float]]] = []
    for doc_id, similar_docs in sorted_records_dict.items():
        for doc_id_2, sw_score_phrase_pair in similar_docs['sw_w'].items():
            if sw_score_phrase_pair[0] >= int(sw_score_threshold):
                if doc_id_2 not in similar_docs['tf_idf']:
                    # do one-off tf-idf
                    doc_1_TF_IDF_vector, doc_2_TF_IDF_vector = get_tiny_TF_IDF_vectors(doc_id, doc_id_2)
                    tf_idf_score = round(1 - fastdist.cosine(doc_1_TF_IDF_vector, doc_2_TF_IDF_vector), 4)
                else:
                    tf_idf_score = similar_docs['tf_idf'][doc_id_2]
                best_results.append({
                    'query_id': doc_id,
                    'doc_id_2': doc_id_2,
                    'sw_w': sw_score_phrase_pair[0],
                    'sw_w_phrase': sw_score_phrase_pair[1],
                    'tf_idf': tf_idf_score,
                    'topic': calculate_topic_similarity_score(doc_id, doc_id_2),
                })
            else:
                break
    # print("organize best:", calc_dur(start3, datetime.now().time()))

    # print("overall:", calc_dur(start0, datetime.now().time()))

    return best_results


def format_batch_results(results, doc_id_1, doc_id_2, priority_texts):
    # calculate number of docs
    batch_size = doc_ids.index(doc_id_2) - doc_ids.index(doc_id_1)

    # begin with head of table
    table_header_HTML = """
                    <h1 align="center">Similarity Results: {} – {} ({} docs)</h1>""".format(
        doc_id_1, doc_id_2, batch_size
    )
    table_header_HTML += "<br>"
    table_header_HTML += """
        <table id="batch_result_table" class="display">
          <thead>
            <tr>
              <th>{}</th>
              <th>{}</th>
              <th>{}</th>
              <th>{}</th>
              <th>{}</th>
              <th>{}</th>
              <th>{}</th>
            </tr>
          </thead>
          <tbody>
    """.format('#',
               'query doc',
               'similar doc',
               'topic score',
               'vocab score',
               'phrase score',
               'full phrase overlap',
               )

    # format rows
    table_rows_HTML = format_batch_result_rows(results, priority_texts)

    # close off table
    table_footer_HTML = """
          </tbody>
        </table>
    """

    docExploreInner_HTML = HTML_templates['docExploreBatchInner'].substitute(
        table_header_HTML=table_header_HTML,
        table_rows_HTML=table_rows_HTML,
        table_footer_HTML=table_footer_HTML,
    )

    return docExploreInner_HTML


def calculate_topic_similarity_score(doc_id_1, doc_id_2):
    doc_1_topic_vector = np.array(thetas[doc_id_1])
    doc_2_topic_vector = np.array(thetas[doc_id_2])
    return round(1-fastdist.cosine(doc_1_topic_vector, doc_2_topic_vector), 3)

def order_results(results):
    return sorted(
        results, key=lambda result: (
            doc_ids.index(result['query_id']), doc_ids.index(result['doc_id_2'])
        )
    )

def format_batch_result_rows(results: List[Dict[str, Union[str, float]]], priority_texts):

    # filter and resort
    results = [r for r in results if parse_complex_doc_id(r['doc_id_2'])[0] in priority_texts]
    results = order_results(results)

    HTML_rows = ""
    for i, result in enumerate(results):
        HTML_rows += """
            <tr>
              <td>{}</td>
              <td>{}</td>
              <td>{}</td>
              <td>{:.1%}</td>
              <td>{:.1%}</td>
              <td>{}</td>
              <td>{}</td>
            </tr>
        """.format(
            i+1,
            format_docView_link(result['query_id']),
            format_docView_link(result['doc_id_2']),
            result['topic'],
            result['tf_idf'],
            result['sw_w'],
            format_docCompare_link(result['query_id'], result['doc_id_2'], result['sw_w_phrase'], title=""),
        )
    return HTML_rows


def score_to_color(score):
    alpha = score # both [0,1]
    color = (165, 204, 107, alpha) # this is a nice green
    return str(color) # return tuple with parentheses

def compare_readings(reading_A, reading_B):

    score = SequenceMatcher(a=reading_A, b=reading_B).ratio()
    score = int(score*100)/100 # hard round to two decimal places
    if   (0.00 <= score <= 0.25): return 0.0
    elif (0.25 < score <= 0.50): return 0.2
    elif (0.50 < score <= 0.75): return 0.4
    elif (0.75 < score <= 1.00): return 0.7
    else: import pdb; pdb.set_trace()


    # for comparison: trivially by len
    # diff = abs(len(reading_A) - len(reading_B))
    # penalty = diff / max(len(reading_A), len(reading_B))
    # score = int((1 - penalty)*100)/100 # [0,1] where 1 is more similar
    # if   (0.00 < score <= 0.25): return 0.2
    # elif (0.25 < score <= 0.50): return 0.4
    # elif (0.50 < score <= 0.75): return 0.5
    # elif (0.75 < score <= 1.00): return 0.6


def remove_stopwords(reading):
    reading_words = reading.split(' ') # explicit so as to preserve initial and final spaces
    return ' '.join( [ word for word in reading_words
                        if word not in stopwords ] )

def nw_align(text_1, text_2):

    # using CollateX algorithm (actually not exactly NW)

    num_score = 0.0

    collation = Collation()
    collation.add_plain_witness("A", text_1)
    collation.add_plain_witness("B", text_2)
    alignment_tei_xml = collate(collation, segmentation=True, near_match=False, output='tei')
    root = etree.fromstring(alignment_tei_xml)

    highlighted_HTML_1 = '<p>'; highlighted_HTML_2 = '<p>'
    highlight_score = 0
    # style = "style='background-color: rgb(255, 255, {})'" # [0,255]
    style = "style='background-color: rgba{}'" # to be formatted with help of score_to_color()

    for node in root.xpath("child::node()"):

        if isinstance(node, etree._ElementUnicodeResult): # a shared reading

            shared_reading = node

            tmp_shared_reading = remove_stopwords(shared_reading)
            if tmp_shared_reading in ["", " "]:
                highlight_score = 0
            else:
                highlight_score = 1
                num_score += len(tmp_shared_reading) * highlight_score
            # highlight_score = 1

            color = score_to_color(highlight_score)
            highlighted_HTML_1 += "<span {}'>{}</span>".format( style.format(color), shared_reading)
            highlighted_HTML_2 += "<span {}'>{}</span>".format( style.format(color), shared_reading)

        elif isinstance(node, etree._Element): # only <app> possible

            num_children = len(node.getchildren()) # either 1 or 2

            if num_children == 1: # one unique reading

                rdg_element = list(node.getchildren())[0]
                unique_reading = rdg_element.xpath("text()")[0]

                highlight_score = 0
                color = score_to_color(highlight_score)

                if rdg_element.get("wit") == "#A":
                    highlighted_HTML_1 += "<span {}'>{}</span>".format( style.format(color), unique_reading)
                elif rdg_element.get("wit") == "#B":
                    highlighted_HTML_2 += "<span {}'>{}</span>".format( style.format(color), unique_reading)

            elif num_children == 2: # two different readings

                rdg_elements = list(node.getchildren())

                text_content_A = rdg_elements[0].xpath("text()")
                text_content_B = rdg_elements[1].xpath("text()")

                reading_A = text_content_A[0] if text_content_A else ""
                reading_B = text_content_B[0] if text_content_B else ""

                tmp_reading_A = remove_stopwords(reading_A)
                tmp_reading_B = remove_stopwords(reading_B)
                if tmp_reading_A in ["", " "]: tmp_reading_A = " "
                if tmp_reading_B in ["", " "]: tmp_reading_B = " "

                highlight_score = compare_readings(tmp_reading_A, tmp_reading_B)
                num_score += len(tmp_reading_A) * highlight_score
                color = score_to_color(highlight_score)

                highlighted_HTML_1 += "<span {}'>{}</span>".format( style.format(color), reading_A)
                highlighted_HTML_2 += "<span {}'>{}</span>".format( style.format(color), reading_B)

    highlighted_HTML_1 += '</p>'; highlighted_HTML_2 += '</p>'

    return highlighted_HTML_1, highlighted_HTML_2, num_score


from radaniba import sw as sw_align

def sw_nw_align(seq1, seq2):
    # returns 2 strings of HTML with color formatted plus numerical score

    # split docs in thirds based on central alignment feature as determined by local sw on char-level

    seq1_pos, seq2_pos, subseq1_len, subseq2_len, score = sw_align(seq1, seq2) # char-level
    if (seq1_pos, seq2_pos, subseq1_len, subseq2_len, score) == (0, 0, 0, 0, 0):
        return "<p>%s</p>" % seq1, "<p>%s</p>" % seq2, 0

    # now do global nw on each pair A-B-C (provided both are non-empty)

    seq1_A, seq1_B, seq1_C = seq1[:seq1_pos], seq1[seq1_pos:seq1_pos+subseq1_len], seq1[seq1_pos+subseq1_len:]
    seq2_A, seq2_B, seq2_C = seq2[:seq2_pos], seq2[seq2_pos:seq2_pos+subseq2_len], seq2[seq2_pos+subseq2_len:]

    if seq1_A == "" or seq2_A == "":
        # i.e., beginning of one lines up with middiLe/end of other
        res1_A, res2_A, score_A = "<p>%s</p>" % seq1_A, "<p>%s</p>" % seq2_A, 0
    else:
        res1_A, res2_A, score_A = nw_align(seq1_A, seq2_A)

    res1_B, res2_B, score_B = nw_align(seq1_B, seq2_B)

    if seq1_C == "" or seq2_C == "":
        res1_C, res2_C, score_C = "<p>%s</p>" % seq1_C, "<p>%s</p>" % seq2_C, 0
    else:
        res1_C, res2_C, score_C = nw_align(seq1_C, seq2_C)

    # piece back together and return

    res1 = res1_A[:-4] + res1_B[3:-4] + res1_C[3:]
    res2 = res2_A[:-4] + res2_B[3:-4] + res2_C[3:]
    full_score =  score_A + score_B + score_C

    return res1, res2, full_score


def compare_doc_pair(   doc_id_1, doc_id_2,
                        topic_labels=topic_interpretations,
                        priority_texts=list(text_abbrev2fn.keys()),
                        N_tf_idf=search_N_defaults["N_tf_idf"],
                        N_sw_w=search_N_defaults["N_sw_w"],
                        similarity_data: Optional[PymongoCollection]=None,
                        ):

    # start0 = datetime.now().time()

    text_1, text_2 = doc_fulltext[doc_id_1], doc_fulltext[doc_id_2]

    query = {"query_id": doc_id_1}
    record = similarity_data.find_one(query)
    similar_docs = record['similar_docs']

    # do one-off topic comparison
    # start1 = datetime.now().time()

    doc_1_topic_vector = np.array(thetas[doc_id_1])
    doc_2_topic_vector = np.array(thetas[doc_id_2])
    topic_similiarity_score = round(1-fastdist.cosine(doc_1_topic_vector, doc_2_topic_vector), 4)

    # print("do one-off topic comparison:", calc_dur(start1, datetime.now().time()))
    # print("overall:", calc_dur(start0, datetime.now().time()))

    if doc_id_2 in similar_docs['tf_idf']:
        TF_IDF_comparison_score = similar_docs['tf_idf'][doc_id_2]
    else:
        # do one-off tf-idf comparison
        # start1 = datetime.now().time()

        doc_1_TF_IDF_vector, doc_2_TF_IDF_vector = get_tiny_TF_IDF_vectors(doc_id_1, doc_id_2)
        TF_IDF_comparison_score = round(1 - fastdist.cosine(doc_1_TF_IDF_vector, doc_2_TF_IDF_vector), 4)

        # print("do one-off tf-idf comparison:", calc_dur(start1, datetime.now().time()))
        # print("overall:", calc_dur(start0, datetime.now().time()))

    if doc_id_2 in similar_docs['sw_w']:
        sw_w_align_score = similar_docs['sw_w'][doc_id_2][0]
    else:
        # do one-off sw_w comparison
        # start1 = datetime.now().time()

        subseq1_pos, subseq2_pos, subseq1_len, subseq2_len, score = sw_align(text_1, text_2, words=True)
        if (subseq1_pos, subseq2_pos, subseq1_len, subseq2_len, score) == (0, 0, 0, 0, 0):
            sw_w_align_score = 0
        else:
            subseq1 = ' '.join(text_1.split(' ')[subseq1_pos:subseq1_pos + subseq1_len])
            subseq2 = ' '.join(text_2.split(' ')[subseq2_pos:subseq2_pos + subseq2_len])
            _, _, _, _, score = sw_align(subseq1, subseq2, words=False)
            sw_w_align_score = str(score / 10)
        # print("# do one-off sw_w comparison:", calc_dur(start1, datetime.now().time()))
        # print("overall:", calc_dur(start0, datetime.now().time()))

    # do actual overall alignment
    # start1 = datetime.now().time()

    highlighted_HTML_1, highlighted_HTML_2, score = sw_nw_align(text_1, text_2)
    # sw_nw_score = "{:.1f}".format(score)  # former "Composite Alignment Score": "char-level, Needleman-Wunsch primed with Smith-Waterman, number of chars highlighted dark green"

    # print("do actual overall alignment:", calc_dur(start1, datetime.now().time()))
    # print("overall:", calc_dur(start0, datetime.now().time()))

    # also prepare similar_doc_links
    # start1 = datetime.now().time()

    common_kwargs = {
        "topic_labels": topic_labels,
        "priority_texts": priority_texts,
        "N_tf_idf": N_tf_idf,
        "N_sw_w": N_sw_w,
        "results_as_links_only": True,
        "similarity_data": similarity_data,
    }
    similar_doc_links_for_1 = get_closest_docs(doc_id_1, **common_kwargs)
    similar_doc_links_for_2 = get_closest_docs(doc_id_2, **common_kwargs)

    # print("prepare similar_doc_links:", calc_dur(start1, datetime.now().time()))
    # print("overall:", calc_dur(start0, datetime.now().time()))

    # make similar doc buttons show up and populate
    # also anticipate needing numerical position in (ordered) dict (see index() below)
    # start1 = datetime.now().time()

    if doc_id_2 in similar_doc_links_for_1: # then want buttons to show up on right
        activate_similar_link_buttons_right = 1
        ks_1 = list(similar_doc_links_for_1.keys())
        prev_sim_doc_id_for_1 = similar_doc_links_for_1[doc_id_2]['prev']
        next_sim_doc_id_for_1 = similar_doc_links_for_1[doc_id_2]['next']
        sim_rank_of_prev_for_1 = ks_1.index(prev_sim_doc_id_for_1) + 1 if prev_sim_doc_id_for_1 else None
        sim_rank_of_2_for_1 = ks_1.index(doc_id_2) + 1
        sim_rank_of_next_for_1 = ks_1.index(next_sim_doc_id_for_1) + 1 if next_sim_doc_id_for_1 else None
    else:
        activate_similar_link_buttons_right = ""
        prev_sim_doc_id_for_1 = next_sim_doc_id_for_1 = sim_rank_of_prev_for_1 = sim_rank_of_2_for_1 = sim_rank_of_next_for_1 = ""

    if doc_id_1 in similar_doc_links_for_2: # then want buttons to show up on left
        activate_similar_link_buttons_left = 1
        ks_2 = list(similar_doc_links_for_2.keys())
        prev_sim_doc_id_for_2 = similar_doc_links_for_2[doc_id_1]['prev']
        next_sim_doc_id_for_2 = similar_doc_links_for_2[doc_id_1]['next']
        sim_rank_of_prev_for_2 = ks_2.index(prev_sim_doc_id_for_2) + 1 if prev_sim_doc_id_for_2 else None
        sim_rank_of_1_for_2 = ks_2.index(doc_id_1) + 1
        sim_rank_of_next_for_2 = ks_2.index(next_sim_doc_id_for_2) + 1 if next_sim_doc_id_for_2 else None
    else:
        activate_similar_link_buttons_left = ""
        prev_sim_doc_id_for_2 = next_sim_doc_id_for_2 = sim_rank_of_prev_for_2 = sim_rank_of_1_for_2 = sim_rank_of_next_for_2 = ""

    # print("make similar doc buttons show up and populate:", calc_dur(start1, datetime.now().time()))
    # print("overall:", calc_dur(start0, datetime.now().time()))

    # format HTML results
    # start1 = datetime.now().time()

    results_HTML = HTML_templates['docCompareInner'].substitute(
                    doc_id_1=doc_id_1, doc_id_2=doc_id_2,
                    doc_id_1_work_name=(doc_id_1_work_name := parse_complex_doc_id(doc_id_1)[0]),
                    doc_id_2_work_name=(doc_id_2_work_name := parse_complex_doc_id(doc_id_2)[0]),
                    doc_id_1_local=(doc_id_1_local := get_full_local_doc_id(doc_id_1)),
                    doc_id_2_local=(doc_id_2_local := get_full_local_doc_id(doc_id_2)),
                    text_1_display_name=clean_titles[doc_id_1_work_name],
                    text_2_display_name=clean_titles[doc_id_2_work_name],

                    text_1_doc_pos=abbrv2docs[doc_id_1_work_name].index(doc_id_1_local)+1,
                    text_2_doc_pos=abbrv2docs[doc_id_2_work_name].index(doc_id_2_local)+1,
                    text_1_doc_count=len(abbrv2docs[doc_id_1_work_name]),
                    text_2_doc_count=len(abbrv2docs[doc_id_2_work_name]),

                    doc_section_1=section_labels[doc_id_1],
                    doc_section_2=section_labels[doc_id_2],

                    first_doc_id_1=get_full_local_doc_id(doc_links[doc_id_1]['first']),
                    first_doc_id_2=get_full_local_doc_id(doc_links[doc_id_2]['first']),
                    prev_doc_id_1=get_full_local_doc_id(doc_links[doc_id_1]['prev']),
                    prev_doc_id_2=get_full_local_doc_id(doc_links[doc_id_2]['prev']),
                    next_doc_id_1=get_full_local_doc_id(doc_links[doc_id_1]['next']),
                    next_doc_id_2=get_full_local_doc_id(doc_links[doc_id_2]['next']),
                    last_doc_id_1=get_full_local_doc_id(doc_links[doc_id_1]['last']),
                    last_doc_id_2=get_full_local_doc_id(doc_links[doc_id_2]['last']),

                    prev_sim_doc_id_for_2=prev_sim_doc_id_for_2, # left
                    next_sim_doc_id_for_2=next_sim_doc_id_for_2,
                    sim_rank_of_prev_for_2=sim_rank_of_prev_for_2,
                    sim_rank_of_1_for_2=sim_rank_of_1_for_2,
                    sim_rank_of_next_for_2=sim_rank_of_next_for_2,

                    prev_sim_doc_id_for_1=prev_sim_doc_id_for_1, # right
                    next_sim_doc_id_for_1=next_sim_doc_id_for_1,
                    sim_rank_of_prev_for_1=sim_rank_of_prev_for_1,
                    sim_rank_of_2_for_1=sim_rank_of_2_for_1,
                    sim_rank_of_next_for_1=sim_rank_of_next_for_1,

                    N_sw_w=N_sw_w,

                    doc_segmented_highlighted_fulltext_1=highlighted_HTML_1,
                    doc_segmented_highlighted_fulltext_2=highlighted_HTML_2,
                    top_topics_summary_1=format_top_topic_summary(
                        doc_id_1,
                        get_top_topic_indices(doc_id_1, max_N=5, threshold=0.03),
                        topic_labels=topic_labels
                        ),
                    top_topics_summary_2=format_top_topic_summary(
                        doc_id_2,
                        get_top_topic_indices(doc_id_2, max_N=5, threshold=0.03),
                        topic_labels=topic_labels
                        ),

                    topic_similiarity_score=round(topic_similiarity_score,2),
                    TF_IDF_comparison_score=round(TF_IDF_comparison_score,2),
                    sw_w_align_score=sw_w_align_score,
                    )

    # print("format HTML results:", calc_dur(start1, datetime.now().time()))
    # print("overall:", calc_dur(start0, datetime.now().time()))

    return results_HTML, activate_similar_link_buttons_left, activate_similar_link_buttons_right



def format_topic_adjust_output(topic_label_input):

    overall_buffer = ""
    for i, label in enumerate(topic_label_input):

        topic_row_buffer = """
<div class='row'>"""

        # add topic_explore_links and label edit field
        topic_row_buffer += """
    <div class="col-md-1">
        <p><big>{}</big></p>
    </div>""".format(format_topic_explore_links(i))
        topic_row_buffer += """
    <div class="col-md-4">
        <input id="topic_label_{}" name="topic_label_{}" type="text" class="form-control" value="{}" size="30"/>
    </div>
    <div class="col-md-2"></div>""".format(i+1, i+1, label)

        topic_row_buffer += """
</div><!-- topic row -->"""

        overall_buffer += topic_row_buffer

    topicAdjustInner_HTML = HTML_templates['topicAdjustInner'].substitute(
                                label_HTML=overall_buffer
                                )
    return topicAdjustInner_HTML

num_docs_by_text = {}
for txt_abbrv in list(text_abbrev2fn.keys()):
    num_docs_by_text[txt_abbrv] = len([ doc_id
                                for doc_id in doc_ids
                                if parse_complex_doc_id(doc_id)[0] == txt_abbrv
                           ])


def format_text_prioritize_output(*priority_texts_input):

    overall_buffer = ""
    for abbrev, title in text_abbrev2title.items():

        # doing with templating
        checked_string = "checked" * (abbrev in priority_texts_input)
        overall_buffer += """
        <div class="row">
            <div class='col-md-1'>
                <input type='checkbox' id="checkbox_{}" name="priority_checkboxes" value="{}" {}/>
            </div>
            <div class='col-md-1'>
                <p>{}</p>
            </div>
            <div class='col-md-3'>
                <p>{}</p>
            </div>
            <div class='col-md-2'>
                <p>[{}]</p>
            </div>
        </div>
        """.format(abbrev, abbrev, checked_string, abbrev, title, num_docs_by_text[abbrev])

    # get num of docs in priority_texts to use for computation time calculations
    num_priority_docs = sum([ num_docs_by_text[text_name] for text_name in priority_texts_input ])

    overall_buffer += """
    <div class="row">
        <div class='col-md-1'>
        </div>
        <div class='col-md-1'>
        </div>
        <div class='col-md-3'>
            <p><b>Total number of prioritized documents:</b></p>
        </div>
        <div class='col-md-2'>
            <p>[{}]</p>
        </div>
    </div>
    """.format(num_priority_docs)

    textPrioritizeInner_HTML = HTML_templates['textPrioritizeInner'].substitute(
                                    text_priority_HTML=overall_buffer
                                    )

    return textPrioritizeInner_HTML


def format_search_depth_slider_pair(N_tf_idf, N_sw_w, priority_texts):

    # get num of docs in priority_texts to use for computation time calculations
    num_priority_docs = sum([ num_docs_by_text[text_name] for text_name in priority_texts ])

    N_vals = {
        'N_tf_idf': N_tf_idf,
        'N_sw_w': N_sw_w,
    }

    N_max_vals = {
        'N_tf_idf': num_priority_docs,
        'N_sw_w': N_tf_idf,
    }

    HTML_buffer = """
<div class='row'><!-- topic no-slider -->
    <div class='col-md-2'>
       <p><b>Topics</b></p>
    </div>
    <div class='col-md-6'>
       <p>(Always compared for all docs.)</p>
    </div>
    <div class="col-md-4">
       <p>({} or 100% of docs) * ( {:.7f} s / topic comparison) = {:.2f} s</p>
    </div>
</div><!-- topic no-slider -->
<div class='row'><!-- note no-slider -->
    <div class='col-md-8'>
       <p>(The below two comparisons are performed only for max <a href='textPrioritize'>{} priority docs</a>.)</p>
    </div>
    <div class="col-md-4">
       <p></p>
    </div>
</div><!-- note no-slider -->
""".format(num_docs, topic_secs_per_comparison, num_docs*topic_secs_per_comparison, num_priority_docs)

    slider_JS_buffer = """
<script>"""

    for (long_name, name) in [
        ('TF-IDF', 'N_tf_idf'),
        ('Smith-Waterman', 'N_sw_w'),
    ]:

        simple_name = name[2:]  # tf_idf or sw_w

        row_buffer = """
<div class='row'><!-- slider with text -->
    <div class='col-md-2'>
       <p><b>{}</b></p>
    </div>
    <div class='col-md-6'>
       <div class='range'>
           <input type='range' class='form-range' name='{}_slider' id='{}_slider' min='0' max='{}' step='{}' value='{}'/>
       </div>
    </div>
    <div class="col-md-4">
       <p id="{}_slider_curr_val_p"></p>
    </div>
</div><!-- slider with text -->""".format(long_name, name, name,
                               N_max_vals[name], 1, N_vals[name],
                               name
                               )

        slider_JS_buffer += """
var {}_slider = document.getElementById("{}_slider");
var {}_slider_curr_val_p = document.getElementById("{}_slider_curr_val_p");
{}_computation_time = (Math.round( {}_secs_per_comparison * parseInt({}_slider.value ) * 100) / 100).toFixed(2);
{}_slider_curr_val_p.innerHTML = `(${{ {}_slider.value }} or ${{ (Math.round( parseInt({}_slider.value) / {} * 10000) / 100).toFixed(1) }}% of docs) * ( ${{ {}_secs_per_comparison }} s / {} comparison) = ${{ {}_computation_time }} s`;
""".format( *[name]*4, *[simple_name]*2, *[name]*4, num_docs, *[simple_name]*3)

        HTML_buffer += row_buffer

    HTML_buffer += """
<div class='row'><!-- row for total -->
    <div class='col-md-2'>
        <p><b>Total Computation Time:</b></p>
    </div>
    <div class='col-md-6'></div>
    <div class="col-md-4">
       <p id="total_computation_time_p"></p>
    </div>
</div><!-- row for total -->
    """

    slider_JS_buffer += """
total_computation_time_p = document.getElementById("total_computation_time_p");
N_tf_idf_slider.oninput = function() {{

    N_tf_idf_slider_curr_val_p.innerHTML = `(${{ this.value }} or ${{ (Math.round( parseInt(this.value) / {} * 10000) / 100).toFixed(1) }}% of docs) * ( ${{ tf_idf_secs_per_comparison }} s / tf_idf comparison) = ${{ tf_idf_computation_time }} s`;
    if (parseInt(N_sw_w_slider.value) > parseInt(this.value)) {{
        sw_w_computation_time = (Math.round( (sw_w_secs_per_comparison * this.value) * 100 ) / 100).toFixed(2);
        N_sw_w_slider_curr_val_p.innerHTML = `(${{ this.value }} or ${{ (Math.round( parseInt(this.value) / {} * 10000) / 100).toFixed(1) }}% of docs) * ( ${{ sw_w_secs_per_comparison }} s / sw_w comparison) = ${{ sw_w_computation_time }} s`;
    }}
    N_sw_w_slider.max = this.value;

    tf_idf_computation_time = (Math.round( (tf_idf_secs_per_comparison * this.value) * 100 ) / 100).toFixed(2);

    total_computation_time = (Math.round( (topic_computation_time + parseFloat(tf_idf_computation_time) + parseFloat(sw_w_computation_time)) * 100 ) / 100).toFixed(2);
    total_computation_time_p.innerHTML = `<b>${{ total_computation_time }} s per query</b>`;

}}

N_sw_w_slider.oninput = function() {{

    N_sw_w_slider_curr_val_p.innerHTML = `(${{ this.value }} or ${{ (Math.round( parseInt(this.value) / {} * 10000) / 100).toFixed(1) }}% of docs) * ( ${{ sw_w_secs_per_comparison }} s / sw_w comparison) = ${{ sw_w_computation_time }} s`;

    sw_w_computation_time = (Math.round( (sw_w_secs_per_comparison * this.value) * 100 ) / 100).toFixed(2);
    total_computation_time = (Math.round( (topic_computation_time + parseFloat(tf_idf_computation_time) + parseFloat(sw_w_computation_time)) * 100 ) / 100).toFixed(2);
    total_computation_time_p.innerHTML = `<b>${{ total_computation_time }} s per query</b>`;

}}

total_computation_time = (Math.round( (topic_computation_time + parseFloat(tf_idf_computation_time) + parseFloat(sw_w_computation_time)) * 100 ) / 100).toFixed(2);
total_computation_time_p = document.getElementById("total_computation_time_p");
total_computation_time_p.innerHTML = `<b>${{ total_computation_time }} s per query</b>`;

""".format(*[num_docs]*3)

    slider_JS_buffer += """
</script>"""

    return HTML_buffer, slider_JS_buffer

def format_search_depth_output(N_tf_idf, N_sw_w, priority_texts):

    JS_preamble = """
<script>

const topic_secs_per_comparison = {:.7f};
const tf_idf_secs_per_comparison = {:.7f};
const sw_w_secs_per_comparison = {:.7f};

const topic_computation_time = {:.7f};

var tf_idf_computation_time;
var sw_w_computation_time;

var total_computation_time;
var total_computation_time_p;

</script>
""".format(topic_secs_per_comparison, tf_idf_secs_per_comparison, sw_w_secs_per_comparison, num_docs*topic_secs_per_comparison)

    slider_HTML, slider_JS = format_search_depth_slider_pair(N_tf_idf, N_sw_w, priority_texts)

    search_N_defaults_HTML = """
    <div class='row'>
        <div class='col-md-2 col-xs-2'>
            <p><b>TF-IDF:</b></p>
        </div>
        <div class='col-md-1 col-xs-1'>
            <p>{}</p>
        </div>
    </div>
    <div class='row'>
        <div class='col-md-2 col-xs-2'>
            <p><b>Smith-Waterman:</b></p>
        </div>
        <div class='col-md-1 col-xs-1'>
            <p>{}</p>
        </div>
    </div>
    """.format(search_N_defaults["N_tf_idf"], search_N_defaults["N_sw_w"])

    searchDepthInner_HTML = HTML_templates['searchDepthInner'].substitute(
        slider_HTML=slider_HTML,
        JS_preamble=JS_preamble,
        slider_JS=slider_JS,
        search_N_defaults=search_N_defaults_HTML
        )
    return searchDepthInner_HTML
