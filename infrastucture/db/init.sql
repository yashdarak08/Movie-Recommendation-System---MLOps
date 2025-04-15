-- Create tables for storing movie recommendations
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    external_user_id INTEGER UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id SERIAL PRIMARY KEY,
    external_movie_id INTEGER UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    genres VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ratings (
    rating_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    movie_id INTEGER REFERENCES movies(movie_id),
    rating FLOAT NOT NULL,
    timestamp BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    movie_id INTEGER REFERENCES movies(movie_id),
    score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX idx_user_movie ON ratings (user_id, movie_id);
CREATE INDEX idx_recommendation_user ON recommendations (user_id);