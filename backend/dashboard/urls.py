from django.urls import path
from .views import register, login, update_user_profile

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', login, name='login'),
    path('update_profile/', update_user_profile, name='update_profile'),
]