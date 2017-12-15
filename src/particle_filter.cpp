/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random> // used for sampling from normal ditributions
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// random engine to be used to genterate Gaussian distributions
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Set the number of particles
  num_particles = 100; 

  // Create a normal distribution arount the first point
  //Generates random numbers according to the Normal (or Gaussian) random number distribution
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // initialize all particles to first position with noise 
  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.id      = i;
    p.x       = dist_x(gen);
    p.y       = dist_y(gen);
    p.theta   = dist_theta(gen);
    p.weight  = 1.0;

    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Create a normal distribution for sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Calculate the new state
  for (int i=0; i<num_particles; i++) {
    if (fabs(yaw_rate) < 0.00001) { // yaw not changed
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta +
						   yaw_rate * delta_t) -
					       sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) -
					       cos(particles[i].theta +
						   yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add sensor noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

    
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i=0; i<observations.size(); i++) { // for each observation

    // current observation
    LandmarkObs obs = observations[i];
    
    // init min distance to landmark as largest possible number
    double min_dist = numeric_limits<double>::max();

    // init the landmark id place holder to something not possible 
    int map_id = -1;

    for (int j=0; j<predicted.size(); j++) { // for each prediction
      
      // current prediction
      LandmarkObs pred = predicted[j];

      // distance between predicted and current landmarks
      double distance = dist(obs.x, obs.y, pred.x, pred.y);

      // closest predicted landmark to the current observed landmark
      if (distance < min_dist) {
	min_dist = distance;
	map_id = pred.id;
      }
    }

    // set observation id to the nearest predicted landmark id
    observations[i].id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i=0; i<num_particles; i++) { // for each particle

    // store x, y, theta
    double p_x   = particles[i].x;
    double p_y   = particles[i].y;
    double p_theta = particles[i].theta;

    // vector to store predicted map landmarks within sensor range of particle
    vector<LandmarkObs> predictions;

    for (int j=0; j<map_landmarks.landmark_list.size(); j++) { // for each landmark
      // store x, y, coordinates and id
      float landmark_x  = map_landmarks.landmark_list[j].x_f;
      float landmark_y  = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      // only wory about landmarks within the range of the sensor
      if (fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y -
							 p_y) <= sensor_range) {
	// add to predictions vector
	predictions.push_back(LandmarkObs{landmark_id, landmark_x,
	      landmark_y});
      }
    }
    // Create a list of for observations transformed to map coordinates
    vector<LandmarkObs> transformed_xy;
    // Transform observation coordinates to map coordinates
    for (int j=0; j<observations.size(); j++) {
      double transformed_x = cos(p_theta) * observations[j].x -
	sin(p_theta) * observations[j].y + p_x;
      double transformed_y = sin(p_theta) * observations[j].x +
	cos(p_theta) * observations[j].y + p_y;
      transformed_xy.push_back(LandmarkObs{observations[j].id,
	    transformed_x, transformed_y});
    }
    // Data association of the predictions and the transformed observations on each particle
    dataAssociation(predictions, transformed_xy);

    // reset weight
    particles[i].weight = 1.0;

    // Calculate weights
    for (int j=0; j<transformed_xy.size(); j++) {
      double o_x = transformed_xy[j].x;
      double o_y = transformed_xy[j].y;

      int landmark_id = transformed_xy[j].id;

      // store xy coordinates of prediction
      double x_p, y_p;
      for (int k=0; k<predictions.size(); k++) {
	if (predictions[k].id == landmark_id) {
	  x_p = predictions[k].x;
	  y_p = predictions[k].y;
	}
      }

      // update weight using multi-variate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = ( 1/(2 * M_PI * s_x * s_y)) * exp(-(pow(x_p - o_x,2) / (2 * pow(s_x,2)) + (pow(y_p - o_y,2) / (2 * pow(s_y,2)))));

      // total obs weight with product of current obs weight
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Store all current weights and find max weight
  vector<double> weights;
  double maxWeight = numeric_limits<double>::min();
  for (int i=0; i<num_particles; i++) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > maxWeight) {
      maxWeight = particles[i].weight;
    }
  }

  // Generate random starting index for resampling wheel
  uniform_int_distribution<int> uni_dist(0, num_particles-1);
  auto index = uni_dist(gen);

  // Uniform random dist from 0.0 to max_weight
  uniform_real_distribution<double> uni_real_dist(0.0, maxWeight);

  double beta = 0.0;

  // Impliment resample wheel
  vector<Particle> resampledParticles;
  for (int i=0; i<num_particles; i++) {
    beta += uni_real_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
