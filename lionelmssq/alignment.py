from copy import deepcopy
from typing import List, Set, Tuple
from itertools import chain


def align_list_explanations(
    self, list_explanations_start, list_explanations_end
) -> Tuple[List[List[Set[str]]], List[List[List[float]]], List[int], List[int]]:
    """
    Aligns the list_explanation_start and list_explanation_end to get the final skeleton sequence!
    """

    def _calculate_intersection(list1, list2):
        intersection = []
        list1_copy = deepcopy(list1)
        list2_copy = deepcopy(list2)

        for l1 in list1:
            for l2 in list2:
                if l1 == l2:
                    if l1 in list1_copy and l2 in list2_copy:
                        intersection.append(l1)
                        list1_copy.remove(l1)
                        list2_copy.remove(l2)
                        break
                    break

        return intersection

    def _align_individual_lists(list_explanation_start, list_explanation_end):
        skeleton_list_explanation = []
        skeleton_seq = []
        seq_idx = 0
        list1_idx = 0
        list2_idx = 0

        start_inner_list = list_explanation_start[list1_idx]
        end_inner_list = list_explanation_end[list2_idx]

        while start_inner_list and end_inner_list:
            if list1_idx < len(list_explanation_start):
                start_inner_list = list_explanation_start[list1_idx]

            if list2_idx < len(list_explanation_end):
                end_inner_list = list_explanation_end[list2_idx]

            while end_inner_list:
                while start_inner_list:
                    # print(f"list1: {list_explanation_start[list1_idx]}")
                    # print(f"list2: {list_explanation_end[list2_idx]}")

                    # Calculate intersection of the two lists:
                    nuc = _calculate_intersection(start_inner_list, end_inner_list)
                    # print(f"nuc: {nuc}")

                    # TODO: Treat the special case where nuc is empty
                    if not nuc:
                        return [], []

                    # Special case where nuc is a list of n times the same nuclotide, e.g: ["A", "A", "A"]
                    if len(nuc) > 1 and len(set(nuc)) == 1:
                        for n in nuc:
                            skeleton_list_explanation.append(list(n))
                    else:
                        skeleton_list_explanation.append(list(nuc))

                    for n in nuc:
                        skeleton_seq.append(set(nuc))
                        seq_idx += 1

                    for n in nuc:
                        if n in start_inner_list:
                            start_inner_list.remove(n)
                        if n in end_inner_list:
                            end_inner_list.remove(n)

                    if not end_inner_list:
                        list2_idx += 1
                        if list2_idx < len(list_explanation_end):
                            end_inner_list = list_explanation_end[list2_idx]
                        else:
                            end_inner_list = []

                list1_idx += 1
                if list1_idx < len(list_explanation_start):
                    start_inner_list = list_explanation_start[list1_idx]
                else:
                    start_inner_list = []

            list2_idx += 1
            if list2_idx < len(list_explanation_end):
                end_inner_list = list_explanation_end[list2_idx]
            else:
                end_inner_list = []

        # print(f"Skeleton_list_explanation: {skeleton_list_explanation}")
        # print(f"Skeleton_Seq: {skeleton_seq}")

        return skeleton_seq, skeleton_list_explanation

    start_seq_index = []
    end_seq_index = []
    skeleton_seq = []
    skeleton_list_explanation = []

    for idx_1, seq_1 in enumerate(list_explanations_start):
        for idx_2, seq_2 in enumerate(list_explanations_end):
            for subseq_1 in seq_1:
                for subseq_2 in seq_2:
                    # Do not try to align sequences if there lengths don't match!
                    if len(list(chain.from_iterable(subseq_1))) != len(
                        list(chain.from_iterable(subseq_2))
                    ):
                        continue

                    skeleton_seq_temp, skeleton_list_explanation_temp = (
                        _align_individual_lists(
                            deepcopy(subseq_1), deepcopy(subseq_2)[::-1]
                        )
                    )

                    if skeleton_seq_temp and skeleton_list_explanation_temp:
                        # print(f"Index_1: {idx_1}", f"Index_2: {idx_2}")
                        # print(
                        # f"Skeleton_list_explanation: {skeleton_list_explanation_temp}"
                        # )

                        skeleton_seq.append(skeleton_seq_temp)
                        skeleton_list_explanation.append(skeleton_list_explanation_temp)
                        start_seq_index.append(idx_1)
                        end_seq_index.append(idx_2)

    return skeleton_seq, skeleton_list_explanation, start_seq_index, end_seq_index


def align_skeletons_multi_seq(
    self,
    skeleton_seq_start,
    skeleton_seq_end,
    score_seq_start=None,
    score_seq_end=None,
    nucleosides={"A", "U", "C", "G"},
) -> Tuple[List[List[Set[str]]], List[float], List[int], List[int]]:
    """
    Aligns the skeleton_seq_start and skeleton_seq_end position wise to get the final skeleton sequence!
    If and when there is a disagreement, the nucleotide from the sequence closer to the respective end is preferrentially considered!
    This additionally propagates the scores of the sequences using a formula to get the final alignemnt score based on how much disagreement there is!
    """

    # While building the ladder it may happen that things are unambiguous from one side, but not from the other!
    # In that case, we should consider the unambiguous side as the correct one! If the intersection is empty, then we can consider the union of the two!

    skeleton_seq = []
    skeleton_seq_score = []
    start_seq_index = []
    end_seq_index = []

    skeleton_seq_imperfect = []
    skeleton_seq_score_imperfect = []
    start_seq_index_imperfect = []
    end_seq_index_imperfect = []

    for idx_1, seq_1 in enumerate(skeleton_seq_start):
        for idx_2, seq_2 in enumerate(skeleton_seq_end):
            temp_seq = [set() for _ in range(self.seq_len)]
            perfect_match = True
            temp_score = 0.0

            # print("seq_1 = ", seq_1)
            for i in range(self.seq_len):
                temp_seq[i] = seq_1[i].intersection(seq_2[i])
                if temp_seq[i]:
                    if score_seq_start is not None and score_seq_end is not None:
                        # temp_score += float(len(temp_seq[i]))/(
                        #     score_seq_start[idx_1] + score_seq_end[idx_2]
                        # )*float(min(i,abs(i-self.seq_len)))/self.seq_len
                        temp_score += float(len(nucleosides) - len(temp_seq[i])) * (
                            score_seq_start[idx_1] + score_seq_end[idx_2]
                        )
                    else:
                        # temp_score += float(len(temp_seq[i]))*float(min(i,abs(i-self.seq_len)))/self.seq_len
                        temp_score += float(len(nucleosides) - len(temp_seq[i]))
                else:
                    if score_seq_start is not None and score_seq_end is not None:
                        # temp_score += float(len(nucleosides))/(
                        #     score_seq_start[idx_1] + score_seq_end[idx_2]
                        temp_score += (
                            float(len(nucleosides))
                            * (score_seq_start[idx_1] + score_seq_end[idx_2])
                        )  # Penalize this by the number of nucelosides being considered!
                    else:
                        temp_score += float(
                            len(nucleosides)
                        )  # Penalize this by the number of nucelosides being considered!

                    if len(seq_1[i]) < len(seq_2[i]):
                        temp_seq[i] = seq_1[i]
                    elif len(seq_1[i]) > len(seq_2[i]):
                        temp_seq[i] = seq_2[i]
                    else:
                        if i < self.seq_len / 2:
                            temp_seq[i] = seq_1[i]
                        else:
                            temp_seq[i] = seq_2[i]

                    perfect_match = False

            if perfect_match and temp_seq not in skeleton_seq:
                # print("Perfect match seq = ", temp_seq, "with score = ", temp_score)
                skeleton_seq.append(temp_seq)
                skeleton_seq_score.append(temp_score)
                start_seq_index.append(idx_1)
                end_seq_index.append(idx_2)

            if not perfect_match and temp_seq not in skeleton_seq:
                skeleton_seq_imperfect.append(temp_seq)
                skeleton_seq_score_imperfect.append(temp_score)
                start_seq_index_imperfect.append(idx_1)
                end_seq_index_imperfect.append(idx_2)

    if skeleton_seq:
        sorted_skeleton_seq = [
            seq
            for _, seq in sorted(zip(skeleton_seq_score, skeleton_seq), reverse=True)
        ]
        sorted_start_seq_index = [
            index
            for _, index in sorted(
                zip(skeleton_seq_score, start_seq_index), reverse=True
            )
        ]
        sorted_end_seq_index = [
            index
            for _, index in sorted(zip(skeleton_seq_score, end_seq_index), reverse=True)
        ]
        sorted_skeleton_seq_score = sorted(skeleton_seq_score, reverse=True)

    elif skeleton_seq_imperfect:
        print("No perfect match found, using imperfect matches!")

        sorted_skeleton_seq = [
            seq
            for _, seq in sorted(
                zip(skeleton_seq_score_imperfect, skeleton_seq_imperfect),
                reverse=True,
            )
        ]
        sorted_start_seq_index = [
            index
            for _, index in sorted(
                zip(skeleton_seq_score_imperfect, start_seq_index_imperfect),
                reverse=True,
            )
        ]
        sorted_end_seq_index = [
            index
            for _, index in sorted(
                zip(skeleton_seq_score_imperfect, end_seq_index_imperfect),
                reverse=True,
            )
        ]
        sorted_skeleton_seq_score = sorted(skeleton_seq_score_imperfect, reverse=True)

    return (
        sorted_skeleton_seq,
        sorted_skeleton_seq_score,
        sorted_start_seq_index,
        sorted_end_seq_index,
    )


def align_skeletons(
    self,
    skeleton_seq_start,
    skeleton_seq_end,
    align_depth=None,
    trust_range=None,
    trust_smaller_set=False,
) -> List[Set[str]]:
    """
    Old function to align the skeleton_seq_start and skeleton_seq_end position wise to get the final skeleton sequence!
    Implements various ideas to get the final skeleton sequence!
    Could be useful in the future and hence retained!
    """

    # While building the ladder it may happen that things are unambiguous from one side, but not from the other!
    # In that case, we should consider the unambiguous side as the correct one! If the intersection is empty, then we can consider the union of the two!

    # Align the skeletons of the start and end fragments to get the final skeleton sequence!
    # Wherever there is no ambiguity, that nucleotide is preferrentially considered!

    skeleton_seq = [set() for _ in range(self.seq_len)]

    if align_depth is None:
        if trust_range is None:
            for i in range(self.seq_len):
                skeleton_seq[i] = skeleton_seq_start[i].intersection(
                    skeleton_seq_end[i]
                )
                if not skeleton_seq[i]:
                    if not trust_smaller_set:
                        skeleton_seq[i] = skeleton_seq_start[i].union(
                            skeleton_seq_end[i]
                        )
                    else:
                        if len(skeleton_seq_start[i]) < len(skeleton_seq_end[i]):
                            skeleton_seq[i] = skeleton_seq_start[i]
                        else:
                            skeleton_seq[i] = skeleton_seq_end[i]
        else:
            trust_range = int(self.seq_len / 2)  # TODO: update this algo smartly!
            for i in range(trust_range):
                skeleton_seq[i] = skeleton_seq_start[i]
            for i in range(trust_range, self.seq_len):
                skeleton_seq[i] = skeleton_seq_end[i]
    else:
        for i in range(align_depth):
            skeleton_seq[i] = skeleton_seq_start[i].intersection(skeleton_seq_end[i])
            if not skeleton_seq[i]:
                skeleton_seq[i] = skeleton_seq_start[i].union(skeleton_seq_end[i])
        # for i in range(align_depth, self.seq_len):
        for i in range(-align_depth, 0):
            skeleton_seq[i] = skeleton_seq_start[i].intersection(skeleton_seq_end[i])
            if not skeleton_seq[i]:
                skeleton_seq[i] = skeleton_seq_start[i].union(skeleton_seq_end[i])

    # TODO: Its more complicated, since if two positions are ambigious, they are not indepenedent.
    # If one nucleotide is selected this way, then the same nucleotide cannot be selected in the other position!

    return skeleton_seq
