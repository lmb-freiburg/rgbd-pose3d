from scipy.ndimage.filters import maximum_filter
import scipy.ndimage as ndimage
import numpy as np


def detect_keypoints(scoremap):
    """
    Takes a scoremap and finds locations for keypoints.
    Returns a KxNx2 matrix with the (u, v) coordinates of the N maxima found for the K keypoints.
    """
    assert len(scoremap.shape) == 3, "Needs to be a 3D scoremap."

    keypoint_loc = list()
    for kid in range(scoremap.shape[2]):
        num_kp, maxima = find_maxima(scoremap[:, :, kid])
        if num_kp > 0:
            keypoint_loc.append(maxima)
        else:
            keypoint_loc.append(None)
    return keypoint_loc


def find_maxima(scoremap):
    """
    Takes a scoremap and detect the peaks using the local maximum filter.
    Returns a Nx2 matrix with the (u, v) coordinates of the N maxima found.
    """
    assert len(scoremap.shape) == 2, "Needs to be a 2D scoremap."

    # apply the local maximum filter; all pixel of maximal value
    local_max = maximum_filter(scoremap, size=3)
    mask_max = scoremap == local_max

    # mask out background
    mask_bg = ((np.max(scoremap) - np.min(scoremap)) * 0.25) > scoremap
    mask_max[mask_bg] = False

    # find distinct objects in map
    labeled, num_objects = ndimage.label(mask_max)
    slices = ndimage.find_objects(labeled)

    # create matrix of found objects with their location
    maxima = np.zeros((num_objects, 3), dtype=np.float32)
    for oid, (dy, dx) in enumerate(slices):
        maxima[oid, :2] = [(dx.start + dx.stop - 1)/2, (dy.start + dy.stop - 1)/2]
        u, v = int(maxima[oid, 0] + 0.5), int(maxima[oid, 1] + 0.5)
        maxima[oid, 2] = scoremap[v, u]

    return num_objects, maxima


# At the position limb id it gives you the parent and child keypoint (from where to where the limb goes)
LIMB_KP_PAIRS = [(1, 2), (1, 5),  # Neck -> Shoulder L/R
                 (2, 3), (3, 4),  # Right: Shoulder, Elbow, Wrist
                 (5, 6), (6, 7),  # Left: Shoulder, Elbow, Wrist
                 (1, 8), (8, 9), (9, 10),  # Right: Neck, Hip, Knee, Ankle
                 (1, 11), (11, 12), (12, 13),  # Left: Neck, Hip, Knee, Ankle
                 (1, 0),  # Face: Neck, Nose
                 (0, 14), (14, 16),  # Right: Nose, Eye, Ear
                 (0, 15), (15, 17),  # Left: Nose, Eye, Ear
                 (2, 16), (5, 17)]  # Ear, Shoulder L/R

LIMB_PAF_CHAN = [6, 10, 7, 8, 11, 12, 0, 1, 2, 3, 4, 5, 14, 15, 17, 16, 18, 9, 13]


def calculate_pair_scores(keypoint_det, paf_u, paf_v, min_pairwise_confidence=0.05):
    """ Calculates the matching score from a set of given keypoint detections and the paf's. """
    score_list = list()
    for lid, ((pid, cid), paf_chan) in enumerate(zip(LIMB_KP_PAIRS, LIMB_PAF_CHAN)):
        kp_p = keypoint_det[pid]
        kp_c = keypoint_det[cid]

        score = None
        if (kp_p is not None) and (kp_c is not None):
            # Case when both keypoints are visible
            score = _calc_score(keypoint_det[pid], keypoint_det[cid], paf_u[:, :, paf_chan], paf_v[:, :, paf_chan])

            # sort scores descending (super ugly)
            score = score[score[:, 2].argsort()][::-1, :]

            # discard non optimal ones
            score = _discard_non_optimal_pairwise_terms(score, min_pairwise_confidence)

        score_list.append(score)

    return score_list


def _discard_non_optimal_pairwise_terms(score, min_pairwise_confidence):
    """ Because every keypoint candidate can only be used for a single limb
        its possible to discard many solutions already.
        Also solutions with a low pairwise confidence are removed. """
    # mask = score[:, -1] > min_pairwise_confidence
    # score = score[mask, :]
    indices_p, indices_c, confidences = score[:, 0], score[:, 1], score[:, 2]

    # # get unique indices: This makes every pair unique -> this is not exactly what we want
    # indices_u, ind_selected = np.unique(indices, return_index=True, axis=0)
    # confidences_u = np.expand_dims(confidences[ind_selected], -1)

    # get unique indices of the pid
    _, ind_p_sel = np.unique(indices_p, return_index=True)
    _, ind_c_sel = np.unique(indices_c, return_index=True)

    # create boolean masks from the unique indices
    mask_p = np.zeros(indices_p.shape, dtype='bool')
    mask_c = np.zeros(indices_c.shape, dtype='bool')
    mask_p[ind_p_sel] = True
    mask_c[ind_c_sel] = True

    # merge masks and select subset
    mask_unique = np.logical_and(mask_p, mask_c)
    indices_p_u = indices_p[mask_unique]
    indices_c_u = indices_c[mask_unique]
    confidences_u = confidences[mask_unique]

    return np.stack([indices_p_u, indices_c_u, confidences_u], axis=1)


def _calc_score(coords1, coords2, paf_u, paf_v, step_nums=10, min_score_thresh=0.05, min_score_count=9):
    """ Calculates the score associated with the lines from candidates coord1 to candidates coord2 on the given paf's"""
    score = np.zeros((coords1.shape[0]*coords2.shape[0]+1, 3))  # [i, :] stores [ind1, ind2, score]
    steps = np.linspace(0.0, 1.0, step_nums)

    # iterate all possible combinations of the keypoints forming the limb
    for i1 in range(coords1.shape[0]):
        for i2 in range(coords2.shape[0]):
            # calculate vector spanning the line
            vec = coords2[i2, :2] - coords1[i1, :2]
            norm = np.sqrt(1e-6 + np.sum(np.square(vec)))
            vec_n = vec / norm

            acc_score = 0.0
            acc_count = 0
            for step in steps:
                location = step*vec + coords1[i1, :2]
                u, v = int(location[0] + 0.5), int(location[1] + 0.5)
                tmp_score = vec_n[0]*paf_u[v, u] + vec_n[1]*paf_v[v, u]
                if tmp_score > min_score_thresh:
                    acc_score += tmp_score
                    acc_count += 1

            # set indices
            score[i1*coords2.shape[0]+i2, 0] = i1
            score[i1*coords2.shape[0]+i2, 1] = i2
            if acc_count > min_score_count:
                score[i1*coords2.shape[0]+i2, 2] = acc_score / acc_count

    return score


def group_keypoints(keypoint_det, pairwise_scores, max_num_people=20, num_kp=18, min_num_found_limbs=9, min_score=0.4):
    """ Given lists of detected keypoints and pairwise scores this function detects single persons. """
    # list to accumulate possible solutions
    solutions = np.zeros((max_num_people, num_kp + 2)) # last column is score, second last is counter
    solutions[:, :-2] = -1
    sid = 0

    # iterate limbs
    for lid, (pid, cid) in enumerate(LIMB_KP_PAIRS):
        if pairwise_scores[lid] is None:
            # when there is no pair one or two of the keypoints are not detected
            kp_p = keypoint_det[pid]
            kp_c = keypoint_det[cid]
            if kp_p is not None:
                # parent is available
                score = np.max(kp_p[:, 2])
                kp_ind = np.argmax(kp_p[:, 2])

                # check if its already part of a solution
                if sid > 0:
                    if np.any(kp_ind == solutions[:sid, pid]):
                        break

                # if not create a new solution
                # assert sid < max_num_people-1, "Maximal number of people exceeded."
                if sid < max_num_people-1:
                    solutions[sid, pid] = kp_ind
                    solutions[sid, -1] = score
                    sid += 1

            if kp_c is not None:
                # child is available
                score = np.max(kp_c[:, 2])
                kp_ind = np.argmax(kp_c[:, 2])

                # check if its already part of a solution
                if sid > 0:
                    if np.any(kp_ind == solutions[:sid, cid]):
                        break

                # if not create a new solution
                # assert sid < max_num_people-1, "Maximal number of people exceeded."
                if sid < max_num_people-1:
                    solutions[sid, cid] = kp_ind
                    solutions[sid, -1] = score
                    sid += 1
        else:
            # case when there is actually a pair wise term

            # iterate pair-wise terms
            for score in pairwise_scores[lid]:

                # check if this parent kp is already is part of an existing solution
                if sid > 0:
                    mask = score[0] == solutions[:sid, pid]
                    if np.any(mask):
                        # extend this solution
                        ind = np.where(mask)[0]
                        solutions[ind, cid] = score[1]  # set corresponding child kp
                        solutions[ind, -1] += score[2]  # add score
                        solutions[ind, -2] += 1  # increment limb counter
                        continue

                # create a new solution:
                # assert sid < max_num_people-1, "Maximal number of people exceeded."
                if sid < max_num_people-1:
                    solutions[sid, pid] = score[0]  # set corresponding parent kp
                    solutions[sid, cid] = score[1]  # set corresponding child kp
                    solutions[sid, -1] += score[2]  # add score
                    solutions[sid, -2] += 1  # increment limb counter
                    sid += 1

    # discard unused solution memory
    solutions = solutions[:sid, :]

    # discard bad solutions: 1) minimal number of attached limbs
    num_found_limbs = solutions[:, -2]
    mask = num_found_limbs >= min_num_found_limbs
    solutions = solutions[mask, :]

    # discard bad solutions: 2) score
    solutions[:, -1] /= solutions[:, -2]  # normalize score
    acc_score = solutions[:, -1]
    mask = acc_score >= min_score
    solutions = solutions[mask, :]

    # # sort solutions
    # solutions = solutions[solutions[:, -1].argsort()][::-1, :]

    # assemble solution
    solution_dict = dict()
    for pid, solution in enumerate(solutions):
        solution_dict['person%d' % pid] = dict()
        solution_dict['person%d' % pid]['conf'] = solution[-1]

        solution_dict['person%d' % pid]['kp'] = list()
        for kid in range(solution[:-2].shape[0]):
            candidate_id = int(solution[kid])
            coord = None
            if candidate_id >= 0:
                coord = keypoint_det[kid][candidate_id][:]

            solution_dict['person%d' % pid]['kp'].append(coord)

    return solution_dict



