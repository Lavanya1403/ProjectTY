from django.urls import path
from . import views


# forecast/urls.py
urlpatterns = [
    path('weather/', views.weather_view, name='weather'),
    path('map/', views.map_view, name='map'),
    path('', views.index_view, name='index'),
]