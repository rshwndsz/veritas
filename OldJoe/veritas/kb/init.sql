-- Create role
CREATE ROLE dsouzars WITH LOGIN PASSWORD 'POSTGRES';

-- Create database
CREATE DATABASE veritas WITH OWNER dsouzars;

-- Grant all privileges on database
GRANT ALL PRIVILEGES ON DATABASE veritas TO dsouzars;
