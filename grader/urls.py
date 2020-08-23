from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.question, name='index'),
    path('/essay<int:essay_id>/', views.essay, name='essay'),
]