import argparse
import os
import re
import sys
import unittest
from time import sleep

match = 10
mismatch = -8
gap = -8
# NOTICE NO AFFINE GAP PENALTY, PROBABLY WANT TO ADD THAT


def sw(seq1, seq2, words=False):

    if words:
        # split into words
        seq1 = seq1.split(" ")
        seq2 = seq2.split(" ")

    # extra row and column in scoring matrix for possible starting gap, hence +1
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    # Initialize the scoring matrix.
    score_matrix, (x, y) = create_score_matrix(rows, cols, seq1, seq2)
    if (x, y) == (0, 0):  # no match found whatsoever, likely with words==True
        return 0, 0, 0, 0, 0
    # Traceback. Find the optimal path through the scoring matrix. This path
    # corresponds to the optimal local sequence alignment.
    seq1_aligned, seq2_aligned, delta_x, delta_y = traceback(
        score_matrix, (x, y), seq1, seq2
    )
    assert len(seq1_aligned) == len(
        seq2_aligned
    ), "aligned strings are not the same size"

    # print("starting position in seq1 of optimal alignment:", delta_x)
    # print("original optimally matching segment:", seq1[start_pos[0]-delta_x-1:start_pos[0]-1])
    # print("aligned bits:\n" + seq1_aligned + '\n' + seq2_aligned)
    return x - delta_x, y - delta_y, delta_x, delta_y, score_matrix[x][y]

    # STOP

    # Pretty print the results. The printing follows the format of BLAST results
    # as closely as possible.
    alignment_str, idents, gaps, mismatches = alignment_string(
        seq1_aligned, seq2_aligned
    )
    alength = len(seq1_aligned)
    print()
    print(
        " Identities = {0}/{1} ({2:.1%}), Gaps = {3}/{4} ({5:.1%})".format(
            idents, alength, idents / alength, gaps, alength, gaps / alength
        )
    )
    print()
    for i in range(0, alength, 60):
        seq1_slice = seq1_aligned[i : i + 60]
        print(
            "Query  {0:<4}  {1}  {2:<4}".format(i + 1, seq1_slice, i + len(seq1_slice))
        )
        print("             {0}".format(alignment_str[i : i + 60]))
        seq2_slice = seq2_aligned[i : i + 60]
        print(
            "Sbjct  {0:<4}  {1}  {2:<4}".format(i + 1, seq2_slice, i + len(seq2_slice))
        )
        print()


# def parse_cmd_line():
# 	'''Parse the command line arguments.
# 	Create a help menu, take input from the command line, and validate the
# 	input by ensuring it does not contain invalid characters (i.e. characters
# 	that aren't the bases A, C, G, or T).
# 	'''
# #     seq1 = "ATAGACGACATACAGACAGCATACAGACAGCATACAGA"
# #     seq2 = "TTTAGCATGCGCATATCAGCAATACAGACAGATACG"
# 	seq1 = 'TGTTACGG'
# 	seq2 = 'GGTTGACTA'


def format_matrix(score_matrix, seq1, seq2):
    side_labels = [""] + [unit[:3] for unit in seq1]
    matrix_with_labels = [[side_labels[i]] + row for i, row in enumerate(score_matrix)]

    top_labels = [""] + [unit[:3] for unit in seq2]
    top_labels = [""] + top_labels  # another blank cell for corner
    matrix_with_labels = [top_labels] + matrix_with_labels

    return "\n".join(
        ["\t".join([str(col) for col in row]) for row in matrix_with_labels]
    )


def create_score_matrix(rows, cols, seq1, seq2):
    """Create a matrix of scores representing trial alignments of the two sequences.
    Sequence alignment can be treated as a graph search problem. This function
    creates a graph (2D matrix) of scores, which are based on trial alignments
    of different units. The path with the highest cummulative score is the
    best alignment.
    """
    score_matrix = [[0 for col in range(cols)] for row in range(rows)]
    # maybe can speed this up with numpy.array?
    # Fill the scoring matrix.
    max_score = 0
    max_pos = None  # The row and column of the highest score in matrix.
    for i in range(1, rows):
        for j in range(1, cols):
            score = calc_score(score_matrix, i, j, seq1, seq2)
            if score > max_score:
                max_score = score
                # print("max_score now: ", max_score)
                max_pos = (i, j)
            score_matrix[i][j] = score
        # if i % 5 == 0:
        # 	with open('out.txt','w') as f_out: f_out.write(format_matrix(score_matrix, seq1, seq2))
        # 	sleep(0.2)

    # assert max_pos is not None, 'the x, y position with the highest score was not found'
    if max_pos is None:
        return score_matrix, (0, 0)
        # print('the x, y position with the highest score was not found')
    return score_matrix, max_pos


def calc_score(matrix, x, y, seq1, seq2):
    """Calculate score for a given x, y position in the scoring matrix.
    The score is based on the up, left, and upper-left neighbors.
    """
    similarity = match if seq1[x - 1][:3] == seq2[y - 1][:3] else mismatch
    diag_score = matrix[x - 1][y - 1] + similarity
    up_score = matrix[x - 1][y] + gap
    left_score = matrix[x][y - 1] + gap

    return max(0, diag_score, up_score, left_score)


def traceback(score_matrix, start_pos, seq1, seq2):
    """Find the optimal path through the matrix.
    This function traces a path from the bottom-right to the top-left corner of
    the scoring matrix. Each move corresponds to a match, mismatch, or gap in one
    or both of the sequences being aligned. Moves are determined by the score of
    three adjacent squares: the upper square, the left square, and the diagonal
    upper-left square.
    WHAT EACH MOVE REPRESENTS
            diagonal: match/mismatch
            up:       gap in sequence 1
            left:     gap in sequence 2
    """
    END, DIAG, UP, LEFT = range(4)
    aligned_seq1 = []
    aligned_seq2 = []
    x, y = start_pos
    delta_x = 0
    delta_y = 0
    move = next_move(score_matrix, x, y)
    while move != END:
        if move == DIAG:
            aligned_seq1.append(seq1[x - 1])
            # print("seq1 word appended: ", seq1[x - 1])
            delta_x += 1
            aligned_seq2.append(seq2[y - 1])
            # print("seq2 word appended: ", seq2[y - 1])
            delta_y += 1
            x -= 1
            y -= 1
        elif move == UP:
            aligned_seq1.append(seq1[x - 1])
            # print("seq1 word appended: ", seq1[x - 1])
            aligned_seq2.append("-")
            x -= 1
            delta_x += 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[y - 1])
            # print("seq2 word appended: ", seq2[y - 1])
            y -= 1
            delta_y += 1
        move = next_move(score_matrix, x, y)
    aligned_seq1.append(seq1[x - 1])
    aligned_seq2.append(seq2[y - 1])
    # print("seq1 word appended: ", seq1[x - 1])
    # print("seq2 word appended: ", seq2[y - 1])
    delta_x += 1
    delta_y += 1

    if isinstance(seq1, list):
        return (
            list(reversed(aligned_seq1)),
            list(reversed(aligned_seq2)),
            delta_x,
            delta_y,
        )
    elif isinstance(seq1, str):
        return (
            "".join(reversed(aligned_seq1)),
            "".join(reversed(aligned_seq2)),
            delta_x,
            delta_y,
        )


def next_move(score_matrix, x, y):
    diag = score_matrix[x - 1][y - 1]
    up = score_matrix[x - 1][y]
    left = score_matrix[x][y - 1]
    if diag >= up and diag >= left:  # Tie goes to the DIAG move.
        return 1 if diag != 0 else 0  # 1 signals a DIAG move. 0 signals the end.
    elif up > diag and up >= left:  # Tie goes to UP move.
        return 2 if up != 0 else 0  # UP move or end.
    elif left > diag and left > up:
        return 3 if left != 0 else 0  # LEFT move or end.
    else:
        # Execution should not reach here.
        raise ValueError("invalid move during traceback")


def alignment_string(aligned_seq1, aligned_seq2):
    """Construct a special string showing identities, gaps, and mismatches.
    This string is printed between the two aligned sequences and shows the
    identities (|), gaps (-), and mismatches (:). As the string is constructed,
    it also counts number of identities, gaps, and mismatches and returns the
    counts along with the alignment string.
    AAGGATGCCTCAAATCGATCT-TTTTCTTGG-
    ::||::::::||:|::::::: |:  :||:|   <-- alignment string
    CTGGTACTTGCAGAGAAGGGGGTA--ATTTGG
    """
    # Build the string as a list of characters to avoid costly string
    # concatenation.
    idents, gaps, mismatches = 0, 0, 0
    alignment_string = []
    for base1, base2 in zip(aligned_seq1, aligned_seq2):
        if base1 == base2:
            alignment_string.append("|")
            idents += 1
        elif "-" in (base1, base2):
            alignment_string.append(" ")
            gaps += 1
        else:
            alignment_string.append(":")
            mismatches += 1
    return "".join(alignment_string), idents, gaps, mismatches


def print_matrix(matrix):
    """Print the scoring matrix.
    ex:
    0   0   0   0   0   0
    0   2   1   2   1   2
    0   1   1   1   1   1
    0   0   3   2   3   2
    0   2   2   5   4   5
    0   1   4   4   7   6
    """
    for row in matrix:
        for col in row:
            print("{0:>4}".format(col))
        print()


# class ScoreMatrixTest(unittest.TestCase):
# 	'''Compare the matrix produced by create_score_matrix() with a known matrix.'''
# 	def test_matrix(self):
# 		# From Wikipedia (en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm)
# 		#                -   A   C   A   C   A   C   T   A
# 		known_matrix = [[0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
# 						[0,  2,  1,  2,  1,  2,  1,  0,  2],  # A
# 						[0,  1,  1,  1,  1,  1,  1,  0,  1],  # G
# 						[0,  0,  3,  2,  3,  2,  3,  2,  1],  # C
# 						[0,  2,  2,  5,  4,  5,  4,  3,  4],  # A
# 						[0,  1,  4,  4,  7,  6,  7,  6,  5],  # C
# 						[0,  2,  3,  6,  6,  9,  8,  7,  8],  # A
# 						[0,  1,  4,  5,  8,  8, 11, 10,  9],  # C
# 						[0,  2,  3,  6,  7, 10, 10, 10, 12]]  # A
# 		global seq1, seq2
# 		seq1 = 'AGCACACA'
# 		seq2 = 'ACACACTA'
# 		rows = len(seq1) + 1
# 		cols = len(seq2) + 1
# 		matrix_to_test, max_pos = create_score_matrix(rows, cols)
# 		self.assertEqual(known_matrix, matrix_to_test)

if __name__ == "__main__":
    a = """tatas trayavedane 'para ātmābhyupagantavyaḥ, tasyābhimukhyatrayaṃ svasaṃvedanaṃ ca caturthaṃ prasajyate. punar aparaḥ punar apara iti mahaty anarthaparaṃparā syāt. tasmād ekam eva vedanaṃ tatra bhedāvabhāsa upaplava eva iti. jñānam api svarūpeṇāpratipannam asad eveti śūnyataivāvaśiṣyate. na hi tadanyāpratipatrā tadrūpaparāvṛttaṃ śakyaṃ pratipattum. na cāvedanād vyāvṛttatvānavagamavedanam iti vyavasthāpayituṃ śakyam iti.
—
tad uktam — ""idaṃ vastubalāyātaṃ yad vadanti vipaścitaḥ / yathā yathārthāś cintyante viśīryante tathā tathā //"" iti."""
    b = """saiva tāvat kathaṃ buddhir ekā citrāvabhāsinī // idaṃ vastubalāyātaṃ yad vadanti vipaścitaḥ / yathā yathārthāś cintyante viśīryante tathā tathā // kiṃ syāt sā citrataikasyāṃ na syāt tasyāṃ matāv api / yadīdaṃ svayam arthānāṃ rocate tatra ke vayaṃ // tasmān nārtheṣu na jñāne sthūlābhāsas tadātmanaḥ /"""
    main(a, b)
