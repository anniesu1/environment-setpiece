#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

uniform highp float u_SlowFactor;
uniform vec4 u_Color;

in vec2 fs_Pos;
out vec4 out_Col;

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;

/**
 * Noise functions
 */
float random(vec2 ab) {
	float f = (cos(dot(ab ,vec2(21.9898,78.233))) * 43758.5453);
	return fract(f);
}

float noise(in vec2 xy) {
	vec2 ij = floor(xy);
	vec2 uv = xy-ij;
	uv = uv*uv*(3.0-2.0*uv);
	
	float a = random(vec2(ij.x, ij.y ));
	float b = random(vec2(ij.x+1., ij.y));
	float c = random(vec2(ij.x, ij.y+1.));
	float d = random(vec2(ij.x+1., ij.y+1.));
	float k0 = a;
	float k1 = b-a;
	float k2 = c-a;
	float k3 = a-b-c+d;
	return (k0 + k1*uv.x + k2*uv.y + k3*uv.x*uv.y);
}

vec2 random2( vec2 p , vec2 seed) {
  return fract(sin(vec2(dot(p + seed, vec2(311.7, 127.1)), dot(p + seed, vec2(269.5, 183.3)))) * 85734.3545);
}

float worley(float x, float y, float rows, float cols) {
    float xPos = x * float(rows) / 20.0;
    float yPos = y * float(cols) / 20.0;

    float minDist = 60.0;
    vec2 minVec = vec2(0.0, 0.0);

    // Find closest point
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            vec2 currGrid = vec2(floor(float(xPos)) + float(i), floor(float(yPos)) + float(j));
            vec2 currNoise = currGrid + random2(currGrid, vec2(2.0, 1.0));
            float currDist = distance(vec2(xPos, yPos), currNoise);
            if (currDist <= minDist) {
                minDist = currDist;
                minVec = currNoise;
            }
        }
    }
    return minDist;
}

float bias(float b, float t) {
    return pow(t, log(b) / log(0.5f));
}

float gain(float g, float t) {
    if(t < 0.5f) {
        return bias(1.f-g, 2.f*t) / 2.f;
    } else {
        return 1.f - bias(1.f-g, 2.f - 2.f * t) / 2.f;
    }
}

float falloff(float t) {
    return t*t*t*(t*(t*6.f - 15.f) + 10.f);
}

float lerp(float a, float b, float t) {
    return (1.0 - t) * a + t * b;
}

float dotGridGradient(int ix, int iy, float x, float y, float seed) {
    vec2 dist = vec2(x - float(ix), y - float(iy));
    vec2 rand = (random2(vec2(ix, iy), vec2(seed, seed * 2.139)) * 2.f) - 1.f;
    return dist[0] * rand[0] + dist[1] * rand[1];
}

float perlin(vec2 pos, float seed) {
    //Pixel lies in (x0, y0)
    int x0 = int(floor(pos[0]));
    int x1 = x0 + 1;
    int y0 = int(floor(pos[1]));
    int y1 = y0 + 1;

    float wx = falloff(pos[0] - float(x0));
    float wy = falloff(pos[1] - float(y0));

    float n0, n1, ix0, ix1, value;
    n0 = dotGridGradient(x0, y0, pos[0], pos[1], seed);
    n1 = dotGridGradient(x1, y0, pos[0], pos[1], seed);
    ix0 = lerp(n0, n1, wx);
    n0 = dotGridGradient(x0, y1, pos[0], pos[1], seed);
    n1 = dotGridGradient(x1, y1, pos[0], pos[1], seed);
    ix1 = lerp(n0, n1, wx);
    value = lerp(ix0, ix1, wy);

    return value;
}

float dampen(float t) {
    if(t < 0.4) {
        return pow(t / 0.4, 3.f) * 0.4;
    }
    return t;
}

float fbmPerlin(vec2 pos, float octaves, float seed) {
    float total = 0.f;
    float persistence = 0.5;

    for(float i = 0.f; i < octaves; i++) {
        float freq = pow(2.f, i);
        //divide by 2 so that max is 1
        float amp = pow(persistence, i) / 2.f;
        total += ((perlin(pos * float(freq), seed) + 1.f) / 2.f) * amp;
    }

    return clamp(total, 0.f, 1.f);
}

/**
 * Gradiation
 */
// Mountain palette
const vec3 mountain[5] = vec3[](vec3(3, 6, 13) / 255.0,
                                vec3(14, 32, 72) / 255.0,
                                vec3(69, 88, 121) / 255.0,
                                vec3(117, 134, 163) / 255.0,
                                vec3(114, 171, 198) / 255.0);

const vec3 duskyMountain[5] = vec3[](vec3(223, 196, 182) / 255.0,
                                vec3(233, 122, 144) / 255.0,
                                vec3(128, 71, 102) / 255.0,
                                vec3(54, 94, 122) / 255.0,
                                vec3(249, 250, 252) / 255.0);
vec3 getMountainColor() {
  highp float yPos = 0.5 * (fs_Pos[1] + 1.0);

    if (yPos < 0.05) {
        return mountain[3];
    }
    else if (yPos < 0.1) {
        return mix(mountain[3], mountain[3], (yPos - 0.05) / 0.05);
    }
    else if (yPos < 0.2) {
        return mix(mountain[3], mountain[2], (yPos - 0.1) / .1);
    }
    else if (yPos < 0.4) {
        return mix(mountain[2], mountain[1], (yPos - 0.2) / .2);
    }
    else if (yPos < 0.6) {
        return mix(mountain[1], mountain[0], (yPos - 0.4) / .2);
    }
    return mountain[0];
}

/**
 * Toolbox functions
 */
float triangleWave(float x, float freq, float amplitude) {
  return abs(mod((x * freq), amplitude) - (0.5 * amplitude));
}

/**
 * SDF cominbation operations 
 */ 
float intersectOp(float distA, float distB) {
    return max(distA, distB);
}

float unionOp(float distA, float distB) {
    return min(distA, distB);
}

float differenceOp(float distA, float distB) {
    return max(distA, -distB);
}

/** 
 * Signed distance functions (SDF)
 */

float boxSDF(vec3 p, vec3 boxDim) {
    vec3 d = abs(p) - boxDim;
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    float outsideDistance = length(max(d, 0.0));
    return insideDistance + outsideDistance;
}

float sphereSDF(vec3 p, float r) {
    return length(p) - r;
}

float roundBoxSDF( in vec3 p, in vec3 b, in float r) {
    vec3 q = abs(p) - b;
    return min(max(q.x,max(q.y,q.z)),0.0) + length(max(q,0.0)) - r;
}

float roundConeSDF( in vec3 p, in float r1, float r2, float h ) {
    vec2 q = vec2( length(p.xz), p.y );
    
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
    return dot(q, vec2(a,b) ) - r1;
}

float ellipsoidSDF( in vec3 p, in vec3 r ) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float cylinderSDF(vec3 p, vec2 h) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// arbitrary orientation
float cylinderSDF(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float baba = dot(ba,ba);
    float paba = dot(pa,ba);
#if 0    
    float ibal = inversesqrt(baba);
    float x = length(pa-ba*paba*ibal*ibal) - r;
    float y = (abs(paba-baba*0.5)-baba*0.5)*ibal;
    return min(max(x,y),0.0) + length(max(vec2(x,y),0.0));
#else
    float x = length(pa*baba-ba*paba) - r*baba;
    float y = abs(paba-baba*0.5)-baba*0.5;
    float x2 = x*x;
    float y2 = y*y*baba;
    float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;
#endif    
}

float capsuleSDF(vec3 p, vec3 a, vec3 b, float r) {
	vec3 pa = p-a, ba = b-a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - r;
}

float equilateralTriangleSDF(  in vec2 p ) {
    const float k = 1.73205;//sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x + k*p.y > 0.0 ) p = vec2( p.x - k*p.y, -k*p.x - p.y )/2.0;
    p.x += 2.0 - 2.0*clamp( (p.x+2.0)/2.0, 0.0, 1.0 );
    return -length(p)*sign(p.y);
}

float triPrismSDF( vec3 p, vec2 h ) {
    vec3 q = abs(p);
    float d1 = q.z-h.y;
    h.x *= 0.866025;
    float d2 = equilateralTriangleSDF(p.xy/h.x)*h.x;
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

float coneSDF( in vec3 p, in vec3 c )
{
    vec2 q = vec2( length(p.xz), p.y );
    float d1 = -q.y-c.z;
    float d2 = max( dot(q,c.xy), q.y);
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

float octahedronSDF( in vec3 p, in float s)
{
    p = abs(p);
    float m = p.x+p.y+p.z-s;
    vec3 q;
         if( 3.0*p.x < m ) q = p.xyz;
    else if( 3.0*p.y < m ) q = p.yzx;
    else if( 3.0*p.z < m ) q = p.zxy;
    else return m*0.57735027;
    
    float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
    return length(vec3(q.x,q.y-s+k,q.z-k)); 
}

mat3 rotate2d(float _angle){
    return mat3(cos(_angle),-sin(_angle), 0.0,
                sin(_angle),cos(_angle), 0.0,
                0.0, 0.0, 0.0);
}

vec3 rotateY( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.xz = mat2(co,-si,si,co)*p.xz;
    return p;
}

vec3 rotateX( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.yz = mat2(co,-si,si,co)*p.yz;
    return p;
}
vec3 rotateZ( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.xy = mat2(co,-si,si,co)*p.xy;
    return p;
}

float sceneSDF(vec3 pos, out vec3 col) {
    bool colorSet = false;
    // Central box
    float res = boxSDF(pos + vec3(0.0, 2.5, .0), vec3(0.7, 0.7, 0.7));
    if (res < EPSILON) {
      colorSet = true;
      col = mountain[2];
    }

    res = unionOp(res, sphereSDF(pos + vec3(0.0, .2, 0.0), 0.3)); // head

    vec3 rotate90 = rotateZ(pos + vec3(-1.0, 0.2, .0), -3.15159/ 2.0);
    res = unionOp(res, coneSDF( rotate90, vec3(0.3, 0.1, 0.8) )); // hat

    res = unionOp(res, coneSDF(pos + vec3(0.0, .3, 0.0), vec3(0.3, 0.15, 0.9) )); // body

    res = unionOp(res, capsuleSDF(pos + vec3(-.1, 1.9, 0.0), vec3(0.0,0.,0.), vec3(0.0,1.0,0.0), .03)); // leg
    res = unionOp(res, capsuleSDF(pos + vec3(.1, 1.9, 0.0), vec3(0.0,0.,0.), vec3(0.0,1.0,0.0), .03)); // leg
    if (res < EPSILON && !colorSet) {
      col = vec3(1.0, 1.0, 1.0);
    }

    // Background octahedrons
    res = unionOp(res, octahedronSDF(pos + vec3(7.9, 0.75 + triangleWave(u_Time / u_SlowFactor * noise(fs_Pos), 1.0, 1.0), 0.0), 0.2));
    res = unionOp(res, octahedronSDF(pos + vec3(4.7, 0.0 + triangleWave(u_Time / u_SlowFactor * noise(fs_Pos), 1.0, 1.0), 0.0), 0.2));
    res = unionOp(res, octahedronSDF(pos + vec3(-5.0, 0.75 + triangleWave(u_Time / u_SlowFactor * noise(fs_Pos), 1.0, 1.0), 0.0), 0.4 * noise(fs_Pos)));
    res = unionOp(res, octahedronSDF(pos + vec3(-3.0, 0.75 + triangleWave(u_Time / u_SlowFactor, 1.0, 1.0), 5.0), 0.6));
    res = unionOp(res, octahedronSDF(pos + vec3(2.0, 0.75 + triangleWave(u_Time / u_SlowFactor * 0.4, 1.0, 1.0), 4.0), 0.7 * noise(fs_Pos)));
    if (res < EPSILON && !colorSet) {
      col = vec3(u_Color) / 255.0;
    }
    return res;
}

/**
 * BVH
 */ 
bool didIntersectBoundingBox(vec3 dir, vec3 origin, out float dist, vec3 min, vec3 max) { 
    dir = normalize(dir);
    float tmin = (min.x - origin.x) / dir.x; 
    float tmax = (max.x - origin.x) / dir.x; 

    if (tmin > tmax) {
      // Swap the values
      float temp = tmin;
      tmin = tmax;
      tmax = temp;
    }
    float tymin = (min.y - origin.y) / dir.y; 
    float tymax = (max.y - origin.y) / dir.y; 
    if (tymin > tymax) {
      // Swap the values
      float temp = tymin;
      tymin = tymax;
      tymax = temp;
    }
 
    if ((tmin > tymax) || (tymin > tmax)) {
      return false; 
    }
 
    if (tymin > tmin) {
      tmin = tymin; 
    }
 
    if (tymax < tmax) {
      tmax = tymax; 
    }
 
    float tzmin = (min.z - origin.z) / dir.z; 
    float tzmax = (max.z - origin.z) / dir.z; 
 
    if (tzmin > tzmax) {
      // Swap the values
      float temp = tzmin;
      tzmin = tzmax;
      tzmax = temp;
    }
 
    if ((tmin > tzmax) || (tzmin > tmax)) {
      return false; 
    }
 
    if (tzmin > tmin) {
      tmin = tzmin; 
    } 
 
    if (tzmax < tmax) {
      tmax = tzmax; 
    }
    return true; 
}

// Ray marching
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end, out vec3 col) {
    // BVH
    float dist = 0.0;
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection, col);
        if (dist < EPSILON) {
			    return depth;
        }
        depth += dist;
        if (depth >= end) {
          col = vec3(u_Color);
          return end;
        }
    }
    return end;
}

// Perform a gradient calculation to approximate the normal
vec3 estimateNormal(vec3 p) {
  vec3 col;
  return normalize(vec3(
      sceneSDF(vec3(p.x + EPSILON, p.y, p.z), col) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z), col),
      sceneSDF(vec3(p.x, p.y + EPSILON, p.z), col) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z), col),
      sceneSDF(vec3(p.x, p.y, p.z  + EPSILON), col) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON), col)
  ));
}

// Calculate the phong contribution to light intensity
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
  vec3 N = estimateNormal(p);
  vec3 L = normalize(lightPos - p);
  vec3 V = normalize(eye - p);
  vec3 R = normalize(reflect(-L, N));
  
  float dotLN = dot(L, N);
  float dotRV = dot(R, V);
  
  if (dotLN < 0.0) {
      // Light not visible from this point on the surface
      return vec3(0.0, 0.0, 0.0);
  } 
  if (dotRV < 0.0) {
      // Light reflection in opposite direction as viewer, apply only diffuse component
      return lightIntensity * (k_d * dotLN);
  }
  return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
  // Add ambient light so scene is not totally dark (scale for more/less)
  const vec3 ambientLight = 0.8 * vec3(1.0, 1.0, 1.0);
  vec3 color = ambientLight * k_a;
  
  // Add light 1 on stagr right 
  vec3 light1Pos = vec3(-6.0, -5.0, -6.0 );
  vec3 light1Intensity = vec3(0.5, 0.5, 0.5);
  
  color += phongContribForLight(k_d, k_s, alpha, p, eye, light1Pos, light1Intensity);

  // Add light 2 above the girl
  vec3 light2Pos = vec3(0.0, 
                        10.0,
                        2.0);
  vec3 light2Intensity = vec3(0.9, 0.9, 0.9);
  
  color += phongContribForLight(k_d, k_s, alpha, p, eye, light2Pos, light2Intensity);   

  return color;
}

vec3 castRay() {
  float sx = fs_Pos.x;
  float sy = fs_Pos.y;

  float len = length(u_Ref - u_Eye);
  vec3 look = normalize(u_Ref - u_Eye);
  vec3 right = normalize(cross(look, u_Up));
  vec3 up = cross(right, look);
  float tan_fovy = tan(45.0 / 2.0); // FOV = 45 degrees
  float aspect = u_Dimensions.x / u_Dimensions.y;
  vec3 V = up * len * tan_fovy;
  vec3 H = right * len * aspect * tan_fovy;

  vec3 p = u_Ref + sx * H + sy * V;
  vec3 dir = normalize(p - u_Eye);

  return dir;
}


void main() {
  // Get ray direction
  vec3 dir = castRay();

  // Ray march along ray
  vec3 colorFromScene = vec3(0, 0, 0);

  vec3 col;
  float dist = shortestDistanceToSurface(u_Eye, dir, MIN_DIST, MAX_DIST, col);

  // Lambert's Law for shading
  //vec3 normal = estimateNormal(vec3(fs_Pos, 1.0));
  // float diffuseTerm = dot(normalize(normal), normalize(fs_LightVec));
  // diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);

  if (dist > MAX_DIST - EPSILON) {
    // Stars
    float time = 0.8 * u_Time;
	
	  vec2 position = fs_Pos * 0.5 * u_Dimensions;

	  float color = pow(noise(position), 40.0) * 20.0;

	  float r1 = noise(position*noise(vec2(sin(time*0.01))));
	  float r2 = noise(position*noise(vec2(cos(time*0.01), sin(time*0.01))));
	  float r3 = noise(position*noise(vec2(sin(time*0.05), cos(time*0.05))));
		
	  vec4 starColor = vec4(vec3(color*r1, color*r2, color*r3), 1.0);

    // Sky
    float noise = dampen(gain(0.98, fbmPerlin(fs_Pos + u_Time / 1500.0f, 10.f, 1.328)));
    vec3 pinkClouds = vec3(252.0, 169.0, 184.0) * noise / 255.f;

    // Apply vignette
    vec2 vigPos = vec2(fs_Pos[0], fs_Pos[1]);
    float distance = sqrt((vigPos[0]) * (vigPos[0]) + (vigPos[1]) * (vigPos[1]));
    // Multiply the color by (1 - distance) -- leverage distance of fragment from screen center
    out_Col = (1.0 - distance * 0.4) * (vec4(getMountainColor(), 1.0) + vec4(pinkClouds * 0.3, 0.4) + (starColor * 0.1));
		return;
  } else {
    // Get the closest point on the surface to the eyepoint along the view ray
    vec3 pClosest = u_Eye + dist * dir;
      
    vec3 K_a = vec3(0.2, 0.2, 0.2);
    vec3 K_d = col;
    vec3 K_s = vec3(1.0, 1.0, 1.0);
    float shininess = 10.0;
      
    col = phongIllumination(K_a, K_d, K_s, shininess, pClosest, u_Eye);
    out_Col = vec4(col, 1.0);
    return;
  }
}
