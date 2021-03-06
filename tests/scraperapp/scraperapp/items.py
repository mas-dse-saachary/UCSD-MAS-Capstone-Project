# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScraperappItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    access = scrapy.Field()
    accommodates = scrapy.Field()
    additional_house_rules = scrapy.Field()
    allows_events = scrapy.Field()
    amenities = scrapy.Field()
    bathrooms = scrapy.Field()
    bed_type = scrapy.Field()
    bedrooms = scrapy.Field()
    beds = scrapy.Field()
    calculated_host_listings_count = scrapy.Field()
    calendar_last_scraped = scrapy.Field()
    calendar_updated = scrapy.Field()
    cancellation_policy = scrapy.Field()
    city = scrapy.Field()
    cleaning_fee = scrapy.Field()
    country = scrapy.Field()
    country_code = scrapy.Field()
    description = scrapy.Field()
    experiences_offered = scrapy.Field()
    extra_people = scrapy.Field()
    first_review = scrapy.Field()
    guests_included = scrapy.Field()
    has_availability = scrapy.Field()
    host_about = scrapy.Field()
    host_acceptance_rate = scrapy.Field()
    host_has_profile_pic = scrapy.Field()
    host_id = scrapy.Field()
    host_identity_verified = scrapy.Field()
    host_is_superhost = scrapy.Field()
    host_listings_count = scrapy.Field()
    host_location = scrapy.Field()
    host_name = scrapy.Field()
    host_neighbourhood = scrapy.Field()
    host_picture_url = scrapy.Field()
    host_response_time = scrapy.Field()
    host_since = scrapy.Field()
    host_thumbnail_url = scrapy.Field()
    host_total_listings_count = scrapy.Field()
    host_url = scrapy.Field()
    host_verifications = scrapy.Field()
    house_rules = scrapy.Field()
    id = scrapy.Field()
    interaction = scrapy.Field()
    is_location_exact = scrapy.Field()
    jurisdiction_names = scrapy.Field()
    last_review = scrapy.Field()
    last_scraped = scrapy.Field()
    latitude = scrapy.Field()
    license = scrapy.Field()
    listing_url = scrapy.Field()
    longitude = scrapy.Field()
    market = scrapy.Field()
    maximum_nights = scrapy.Field()
    medium_url = scrapy.Field()
    minimum_nights = scrapy.Field()
    monthly_discount = scrapy.Field()
    monthly_price = scrapy.Field()
    name = scrapy.Field()
    neighborhood_overview = scrapy.Field()
    neighbourhood = scrapy.Field()
    neighbourhood_cleansed = scrapy.Field()
    neighbourhood_group_cleansed = scrapy.Field()
    nightly_price = scrapy.Field()
    notes = scrapy.Field()
    number_of_reviews = scrapy.Field()
    person_capacity = scrapy.Field()
    picture_url = scrapy.Field()
    price = scrapy.Field()
    property_type = scrapy.Field()
    rating_accuracy = scrapy.Field()
    rating_checkin = scrapy.Field()
    rating_cleanliness = scrapy.Field()
    rating_communication = scrapy.Field()
    rating_location = scrapy.Field()
    rating_value = scrapy.Field()
    require_guest_phone_verification = scrapy.Field()
    require_guest_profile_picture = scrapy.Field()
    requires_license = scrapy.Field()
    response_rate = scrapy.Field()
    response_time = scrapy.Field()
    reviews = scrapy.Field()
    review_count = scrapy.Field()
    review_scores_accuracy = scrapy.Field()
    review_scores_checkin = scrapy.Field()
    review_scores_cleanliness = scrapy.Field()
    review_scores_communication = scrapy.Field()
    review_scores_location = scrapy.Field()
    review_scores_rating = scrapy.Field()
    review_scores_value = scrapy.Field()
    reviews_per_month = scrapy.Field()
    room_type = scrapy.Field()
    satisfaction_guest = scrapy.Field()
    scrape_id = scrapy.Field()
    search_price = scrapy.Field()
    security_deposit = scrapy.Field()
    smart_location = scrapy.Field()
    space = scrapy.Field()
    square_feet = scrapy.Field()
    state = scrapy.Field()
    street = scrapy.Field()
    summary = scrapy.Field()
    thumbnail_url = scrapy.Field()
    transit = scrapy.Field()
    url = scrapy.Field()
    weekly_discount = scrapy.Field()
    weekly_price = scrapy.Field()
    xl_picture_url = scrapy.Field()
    zipcode = scrapy.Field()
