import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

# fetch info and format it

data = fetch_movielens(min_rating=4.0)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp')

# train model
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # create recomm for each user we add
    for user_id in user_ids:

        # movies users like already
        known_positive = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model thinks they will love
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("         Known positives:")

        for x in known_positive[:3]:
            print("     %s" % x)

        print("         Recommended:")

        for x in top_items[:3]:
            print("     %s" % x)


sample_recommendation(model, data, [3, 5, 12])
