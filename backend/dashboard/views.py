import re
from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import JsonResponse
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
import logging
from .models import CustomUser

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """Register a new user."""
    data = request.data
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not any([username, email, password]):
        return JsonResponse({
            'error': 'Username, email, and password are required.'
        }, status=status.HTTP_400_BAD_REQUEST)
    if CustomUser.objects.filter(username=username).exists():
        return JsonResponse({
            'error': 'Username already exists.'
        }, status=status.HTTP_400_BAD_REQUEST)
    if CustomUser.objects.filter(email=email).exists():
        return JsonResponse({
            'error': 'Email already exists.'
        }, status=status.HTTP_400_BAD_REQUEST)

    user = CustomUser.objects.create_user(ursername=username, email=email, password=password)    
    user.save()
    return JsonResponse({
        'message': 'User created successfully.'
    }, status=status.HTTP_201_CREATED)
    
@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """Login a user."""
    data = request.data
    
    username = data.get('username')
    if username and re.search(r'.*\..*$', username):
        is_email = True
    else:
        is_email = False
    
    password = data.get('password')
    if is_email:
        user = authenticate(email=username, password=password)
    else:
        user = authenticate(username=username, password=password)
    if user is None:
        return JsonResponse({
            'error': 'Invalid credentials.'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    refresh = RefreshToken.for_user(user)
    return JsonResponse({
        'token': str(refresh.access_token), 
        'refresh': str(refresh),
        'user': user.to_dict(),
    }, status=status.HTTP_200_OK)
       

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_user_profile(request):
    """Update the authenticated user's profile."""
    data = request.data
    user = request.user
    for key, value in data.items():
        if key not in ['address', 'phone_number', 'first_name', 'last_name']:
            logger.error(f"User tried to update invalid field: {key}")
            return JsonResponse({
                'error': f'Invalid field: {key}'
            }, status=status.HTTP_400_BAD_REQUEST)
        setattr(user, key, value)
    user.save()
    return JsonResponse({
        'message': 'User profile updated successfully.'
    }, status=status.HTTP_200_OK)
    