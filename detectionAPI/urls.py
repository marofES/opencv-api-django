from django.urls import path
from .views import *
from rest_framework_simplejwt.views import (
    TokenRefreshView,
)


urlpatterns = [

    path('people-counting/', PeopleDetection.as_view(), name='PeopleDetection'),
    path('vehicle-counting/',VehicleDetection.as_view(), name='VehicleDetection'),
    path('object-detect/',ObjectDetection.as_view(), name='ObjectDetection'),

]