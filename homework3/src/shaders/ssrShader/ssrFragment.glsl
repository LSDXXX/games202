#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWorld;
uniform sampler2D uGShadow;
uniform sampler2D uGPosWorld;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform float uWindowWidth;
uniform float uWindowHeight;

varying mat4 vWorldToScreen;
varying highp vec4 vPosWorld;


#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = uv.x;
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(1.0 - z*z);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = INV_TWO_PI;
  return dir;
}

vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}

void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec4 Project(vec4 a) {
  return a / a.w;
}

float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w;
  return depth;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;
  if (depth < 1e-2) {
    depth = 1000.0;
  }
  return depth;
}

vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;
  return normal;
}

vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;
  return posWorld;
}

float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;
  return visibility;
}

vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec3 wi, vec3 wo, vec2 uv) {
  vec3 diffuse = GetGBufferDiffuse(uv);
  vec3 normal = GetGBufferNormalWorld(uv);
  float c = dot(wi, normal);
  return diffuse*max(c, 0.0)/M_PI;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDirectionalLight(vec2 uv) {
  float shadow = GetGBufferuShadow(uv);
  return uLightRadiance*shadow;
}

bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {

  float maxStep = 500.0;



  vec4 cori = uViewMatrix*vec4(ori, 1.0);
  vec4 cdir = normalize(uViewMatrix*vec4(dir, 0.0));
  vec3 co = cori.xyz/cori.w;
  vec3 cd = cdir.xyz;

  float rayLength = (co+maxStep*cd).z > -1e-3? (-1e-3 - co.z)/cd.z : maxStep;

  vec3 v = ori+rayLength*dir;
  vec4 cvi = uViewMatrix*vec4(v, 1.0);
  vec3 cv = cvi.xyz/cvi.w;

  vec4 so = uProjectionMatrix*vec4(co, 1.0);
  vec4 sv = uProjectionMatrix*vec4(cv, 1.0);
  
  vec2 v0 = so.xy/so.w;
  vec2 v1 = sv.xy/sv.w;

  v1 += (length(v1 - v0) < 0.001)? 0.01:0.0;
  

  vec2 sdelta = v1-v0;
  bool isSwap = false;
  float w = uWindowWidth;
  float h = uWindowHeight;
  if(abs(sdelta.x) < abs(sdelta.y)) {
   sdelta.xy = sdelta.yx;
   w = uWindowHeight;
   h = uWindowWidth;
   isSwap = true;
  }

  float stepDir = sign(sdelta.x);
  float invX = stepDir/(sdelta.x);
  float invW = 2.0/w;
  float invZa = 1.0/so.w, invZb = 1.0/sv.w;
  float dinvZ = (invZb - invZa)*invX*invW;
  vec2 sdir = vec2(stepDir*invW, sdelta.y*invX*invW);

  vec2 inc =vec2(0.0);
  // float invZ = invZa;
  for(float i = 0.0; i < 500.0; i+=1.0) {
    inc += sdir;
    vec2 incuv = isSwap? inc.yx: inc;
    vec2 uv = v0 + incuv; 
    // invZ += dinvZ;
    if(abs(inc.x) > abs(sdelta.x) || uv.x < -1.0 || uv.x > 1.0 || uv.y < -1.0 || uv.y > 1.0) {
      break;
    }
    float s = invX * abs(inc.x);
    float interDepth = 1.0 / (invZa*(1.0-s) + s*invZb);
    // float interDepth = 1.0/invZ;
    float depth = GetGBufferDepth(uv*0.5+0.5);
    if(interDepth > depth+1e-2) {
      hitPos = GetGBufferPosWorld(uv*0.5+0.5);
      // hitPos = vec3(float(i)/500.0);
      return true;
    }
  }

  /*
  float step =0.01;
  vec3 endPoint =ori;
 
  for(int i=0;i<500;i++){
    vec3 testPoint = endPoint+step * dir;
    float testDepth = GetDepth(testPoint);
    vec2 uv = GetScreenCoordinate(testPoint);
    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      break;
    }
    float  bufferDepth = GetGBufferDepth(uv);
    if(step > 40.0){
      return false;
    }else if(testDepth -bufferDepth > 1e-6){
      hitPos = testPoint;
      return true;
    }else if( testDepth < bufferDepth ){
      endPoint =testPoint;
    }else if( testDepth > bufferDepth){
    }

  }
  return false;
  */
  

  return false;
}

#define SAMPLE_NUM 1

void main() {
  float s = InitRand(gl_FragCoord.xy);

  vec3 L = vec3(0.0);

  vec2 uv = GetScreenCoordinate(vPosWorld.xyz);
  // L = GetGBufferDiffuse(GetScreenCoordinate(vPosWorld.xyz));
  vec3 normal = GetGBufferNormalWorld(uv);
  vec3 wi = normalize(vPosWorld.xyz - uCameraPos);
  if(dot(-wi, normal) > 0.0) {
    vec3 wo = normalize(reflect(wi, normal));
    // // L = wo;
    vec3 hitPos = vec3(0.0);
    if(RayMarch(vPosWorld.xyz, wo, hitPos)) {
      L = GetGBufferDiffuse(GetScreenCoordinate(hitPos));
      // L = hitPos;
    //   // L = vec3(1.0);
      // L = hitPos;
      
    }  else {
      // L = GetGBufferDiffuse(uv);

    }
    // vec4 cpos = uProjectionMatrix* uViewMatrix*vPosWorld;
    // wo = (uViewMatrix*vec4(wo, 0.0)).xyz;
    // L = vec3(wo.z);
  }
  // L = GetGBufferDiffuse(uv);
  // vec4 tmp = uViewMatrix*vec4(wi, 0.0);
  // if(tmp.x < 1e-2 && tmp.x > -1e-2 && tmp.y < 1e-2 && tmp.y > -1e-2) {
  //   L = vec3(1.0);
  // }

  
  // L = EvalDiffuse(normalize(uLightDir), vec3(0.0), uv) * EvalDirectionalLight(uv);
  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  // gl_FragColor = vec4(vec3(uWindowHeight/1000.0), 1.0);
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
}
