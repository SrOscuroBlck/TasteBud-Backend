#!/usr/bin/env python3
"""
Reset user account to fresh state for testing onboarding.

This script:
1. Deletes all ratings and interactions
2. Resets taste profile to neutral
3. Clears onboarding completion
4. Allows user to go through onboarding again
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlmodel import Session, select, delete
from config.database import engine
from models import User, Rating, Interaction, OnboardingState
from models.session import RecommendationSession, RecommendationFeedback, PostMealFeedback, UserOrderHistory
from models.interaction_history import UserItemInteractionHistory
from models.bayesian_profile import BayesianTasteProfile
from uuid import UUID
import argparse

# Default user ID
DEFAULT_USER_ID = UUID("385b70e4-35f6-4d8e-9113-d1ef1672afec")


def reset_user(user_id: UUID = None, email: str = None):
    """Reset user to fresh state."""
    
    with Session(engine) as session:
        if email:
            user = session.exec(select(User).where(User.email == email)).first()
            if not user:
                print(f"User with email {email} not found")
                return
            user_id = user.id
        elif user_id:
            user = session.get(User, user_id)
            if not user:
                print(f"User {user_id} not found")
                return
        else:
            user = session.get(User, DEFAULT_USER_ID)
            if not user:
                print(f"Default user {DEFAULT_USER_ID} not found")
                return
            user_id = user.id
        
        print("="*80)
        print("RESETTING USER ACCOUNT")
        print("="*80)
        print(f"User ID: {user_id}")
        print(f"Email: {user.email}")
        print()
        
        # Count existing data
        ratings = session.exec(select(Rating).where(Rating.user_id == user_id)).all()
        interactions = session.exec(select(Interaction).where(Interaction.user_id == user_id)).all()
        onboarding_states = session.exec(select(OnboardingState).where(OnboardingState.user_id == user_id)).all()
        bayesian_profiles = session.exec(select(BayesianTasteProfile).where(BayesianTasteProfile.user_id == user_id)).all()
        rec_sessions = session.exec(select(RecommendationSession).where(RecommendationSession.user_id == user_id)).all()
        order_history = session.exec(select(UserOrderHistory).where(UserOrderHistory.user_id == user_id)).all()
        item_interactions = session.exec(select(UserItemInteractionHistory).where(UserItemInteractionHistory.user_id == user_id)).all()
        
        # Get feedback counts
        feedback_count = 0
        post_meal_count = 0
        for rec_session in rec_sessions:
            feedback_count += len(session.exec(select(RecommendationFeedback).where(RecommendationFeedback.session_id == rec_session.id)).all())
            post_meal_count += len(session.exec(select(PostMealFeedback).where(PostMealFeedback.session_id == rec_session.id)).all())
        
        print(f"Current State:")
        print(f"   Ratings: {len(ratings)}")
        print(f"   Interactions: {len(interactions)}")
        print(f"   Onboarding states: {len(onboarding_states)}")
        print(f"   Bayesian profiles: {len(bayesian_profiles)}")
        print(f"   Recommendation sessions: {len(rec_sessions)}")
        print(f"   Session feedback: {feedback_count}")
        print(f"   Post-meal feedback: {post_meal_count}")
        print(f"   Order history: {len(order_history)}")
        print(f"   Item interactions: {len(item_interactions)}")
        print(f"   Onboarding completed: {user.onboarding_completed}")
        print()
        
        # Delete ratings
        if ratings:
            print(f"️  Deleting {len(ratings)} ratings...")
            for rating in ratings:
                session.delete(rating)
        
        # Delete interactions
        if interactions:
            print(f"️  Deleting {len(interactions)} interactions...")
            for interaction in interactions:
                session.delete(interaction)
        
        # Delete onboarding states
        if onboarding_states:
            print(f"️  Deleting {len(onboarding_states)} onboarding states...")
            for state in onboarding_states:
                session.delete(state)
        
        # Delete Bayesian profiles
        if bayesian_profiles:
            print(f"️  Deleting {len(bayesian_profiles)} Bayesian profiles...")
            for profile in bayesian_profiles:
                session.delete(profile)
        
        # Delete session feedback and post-meal feedback first (foreign key constraint)
        if rec_sessions:
            print(f"️  Deleting session feedback and post-meal feedback...")
            for rec_session in rec_sessions:
                feedbacks = session.exec(select(RecommendationFeedback).where(RecommendationFeedback.session_id == rec_session.id)).all()
                for feedback in feedbacks:
                    session.delete(feedback)
                
                post_meals = session.exec(select(PostMealFeedback).where(PostMealFeedback.session_id == rec_session.id)).all()
                for post_meal in post_meals:
                    session.delete(post_meal)
        
        # Delete recommendation sessions
        if rec_sessions:
            print(f"️  Deleting {len(rec_sessions)} recommendation sessions...")
            for rec_session in rec_sessions:
                session.delete(rec_session)
        
        # Delete order history
        if order_history:
            print(f"️  Deleting {len(order_history)} order history records...")
            for order in order_history:
                session.delete(order)
        
        # Delete item interaction history
        if item_interactions:
            print(f"️  Deleting {len(item_interactions)} item interaction records...")
            for item_interaction in item_interactions:
                session.delete(item_interaction)
        
        # Reset user profile
        print(f"️  Resetting user profile...")
        
        from models.user import TASTE_AXES
        default_axes = TASTE_AXES
        
        user.taste_vector = {axis: 0.5 for axis in default_axes}
        user.taste_uncertainty = {axis: 0.5 for axis in default_axes}
        user.cuisine_affinity = {}
        user.liked_ingredients = []
        user.disliked_ingredients = []
        user.onboarding_completed = False
        user.onboarding_state = {}
        user.onboarding_choices = []
        
        session.add(user)
        session.commit()
        
        print()
        print("="*80)
        print("USER RESET COMPLETE!")
        print("="*80)
        print()
        print("Your account is now completely fresh:")
        print("  All ratings deleted")
        print("  All interactions deleted")
        print("  All Bayesian profiles deleted")
        print("  All recommendation sessions deleted")
        print("  All session feedback deleted")
        print("  All post-meal feedback deleted")
        print("  All order history deleted")
        print("  All item interaction history deleted")
        print("  Taste profile reset to neutral (all 0.5)")
        print("  Onboarding marked as incomplete")
        print("  Ready to test complete flow from scratch")
        print()
        print("Next steps:")
        print("  1. Login to the app")
        print("  2. Complete onboarding (7-10 questions)")
        print("  3. Bayesian profile will be auto-created")
        print("  4. Start a recommendation session")
        print("  5. Get personalized recommendations with new system!")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset user account to fresh state")
    parser.add_argument("--email", type=str, help="User email address")
    parser.add_argument("--user-id", type=str, help="User UUID")
    
    args = parser.parse_args()
    
    user_id = None
    if args.user_id:
        try:
            user_id = UUID(args.user_id)
        except ValueError:
            print(f"Invalid UUID format: {args.user_id}")
            sys.exit(1)
    
    reset_user(user_id=user_id, email=args.email)
