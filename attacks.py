import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier

import numpy as np

def tf_attack(logits_train, logits_test, loss_train, loss_test, train_labels, test_labels):
    attack_input = AttackInputData(
    logits_train = logits_train,
    logits_test = logits_test,
    loss_train = loss_train,
    loss_test = loss_test,
    labels_train = train_labels,
    labels_test = test_labels
    )

    slicing_spec = SlicingSpec(
        entire_dataset = True,
        by_class = False,
        by_percentiles = False,
    by_classification_correctness = False
    )

    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION,
        AttackType.RANDOM_FOREST,
        AttackType.K_NEAREST_NEIGHBORS,
        AttackType.THRESHOLD_ENTROPY_ATTACK,
    ] 

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                    slicing_spec=slicing_spec,
                                 attack_types=attack_types)
    return attacks_result



def simple_mia(sample_loss, members, n_splits=10, random_state=0, classifier = 'LR'):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    
    # Choosing the attack model
    if classifier == 'LR':
        attack_model = linear_model.LogisticRegression()
    elif classifier == 'RF':
        attack_model = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError("attack_model should be either 'LR' or 'RF'")

    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="roc_auc"
    )