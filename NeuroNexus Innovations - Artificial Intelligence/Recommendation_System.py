import pandas as pd
import unittest
from sklearn.metrics.pairwise import cosine_similarity

def get_user_recommendations(user_id, user_item_matrix, user_user_similarity, num_recommendations=5):
    # Filtering the user's ratings and setting item_id as index
    user_ratings = user_item_matrix[user_item_matrix['user_id'] == user_id].set_index('item_id')
  
    # Checking if user_id exists in the user-user similarity matrix
    if user_id in user_user_similarity.index:
        # Finding similar users and sorting by similarity score
        similar_users = user_user_similarity[user_id].sort_values(ascending=False)[1:]
    else:
        similar_users = pd.Series(dtype='float64')  # Assign empty series if user not found

    recommendations = []
    
    unique_item_ids = user_item_matrix['item_id'].unique()
    
    for item_id in unique_item_ids:
        if item_id not in user_ratings.index:
            numerator = 0
            denominator = 0
            for similar_user_id, similarity_score in similar_users.items():
                # Extracting items rated by similar users
                similar_user_items = user_item_matrix[user_item_matrix['user_id'] == similar_user_id]
                common_items = similar_user_items[similar_user_items['item_id'] == item_id]
                if not common_items.empty:  # If common items exist
                    similarity_value = user_user_similarity.loc[user_id, similar_user_id]
                    rating = common_items['rating'].values[0]
                    numerator += similarity_value * rating
                    denominator += similarity_value
            
            if denominator != 0:
                predicted_rating = numerator / denominator
                recommendations.append((item_id, predicted_rating))
                
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_items = [item_id for item_id, _ in recommendations[:num_recommendations]]
    return recommended_items

class TestRecommendationSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a sample user-item matrix and similarity matrix for testing
        data = {
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [101, 102, 101, 103, 102, 104],
            'rating': [5, 4, 3, 1, 2, 5]
        }
        cls.user_item_matrix = pd.DataFrame(data)

        # Generating pivot table and user-user similarity matrix
        pivot_table = cls.user_item_matrix.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        cls.user_user_similarity = pd.DataFrame(cosine_similarity(pivot_table, pivot_table),
                                               index=pivot_table.index, columns=pivot_table.index)
        
    def test_user_item_matrix_conversion(self):
        # Check if the user-item matrix conversion is successful
        self.assertEqual(self.user_user_similarity.shape, (3, 3))

    def test_user_recommendations_exist(self):
        # Test that recommendations exist for each user
        user_ids = self.user_item_matrix['user_id'].unique()
        for user_id in user_ids:
            recommendations = get_user_recommendations(user_id, self.user_item_matrix, self.user_user_similarity)
            self.assertTrue(recommendations)  # Assert that recommendations exist for each user
            
    def test_recommendation_length(self):
        # Test the length of recommendations
        user_id = 1
        num_recommendations = 3
        recommendations = get_user_recommendations(user_id, self.user_item_matrix, self.user_user_similarity, num_recommendations)
        self.assertLessEqual(len(recommendations), num_recommendations)  # Assert the length of recommendations
        
    def test_invalid_user_id(self):
        # Test that no recommendations are returned for an invalid user ID
        invalid_user_id = 100
        recommendations = get_user_recommendations(invalid_user_id, self.user_item_matrix, self.user_user_similarity)
        self.assertEqual(recommendations, [])  # Assert that no recommendations are returned for an invalid user ID

    def test_recommendation_generation(self):
        # Test the generation of recommendations for a specific user
        recommendations = get_user_recommendations(1, self.user_item_matrix, self.user_user_similarity, num_recommendations=2)
        
        # Perform assertions based on expected recommendations
        self.assertIn(103, recommendations)  # Example assertion
        self.assertIn(104, recommendations)  # Example assertion

if __name__ == '__main__':
    unittest.main()
