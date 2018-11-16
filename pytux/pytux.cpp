#ifdef EXECUTABLE
#include "object/camera.hpp"
#include "object/player.hpp"
#include "supertux/game_session.hpp"
#include "supertux/main.hpp"
#include "supertux/sector.hpp"
#include "supertux/screen_manager.hpp"
#include "supertux/screen_fade.hpp"
#include "video/drawing_context.hpp"
#include "controller/basecontroller.h"
#else
#define PYTHON
#endif

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/filesystem.hpp>

#ifdef PYTHON
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#endif

#define MAX_WIDTH (1<<11)
#define MAX_HEIGHT (1<<11)
#define MAX_SIZE ((MAX_WIDTH)*(MAX_HEIGHT))

namespace ip = boost::interprocess;
#ifdef PYTHON
namespace bp = boost::python;
#endif

#ifdef PYTHON
#define NUM_OT 8
#endif

struct TuxSHMem {
	enum StepState {
		STEP_UNKNOWN,
		STEP_START,
		STEP_END,
	};
	// State
	uint8_t synchronized=0, acting=0, restart=0, running=0, quit=0, step_state=0;
	uint32_t seed=0;
	pid_t pid = 0;
	
	// Game state
	uint32_t n_sector = 0;
	float position = 0;
	uint8_t is_winning, is_invincible, is_dying, on_ground, bonus;
	uint16_t coins;
	float velocity[2], bbox[4], camera_pos[2];
	
	// Transferred data
	uint32_t frame_id=0;
	uint32_t action=0, next_action=0;
	uint16_t imW=0, imH=0;
	uint8_t imData[MAX_SIZE*3];
	uint16_t lblW=0, lblH=0;
	uint16_t lblData[MAX_SIZE];
	// Low resolution label data (downsample the region around tux by a factor S to WxH and store a class map)
	uint16_t lrW=0, lrH=0, lrS=0;
	uint8_t lrData[MAX_SIZE];
	uint16_t lrCnt[MAX_SIZE][NUM_OT];
};
struct TuxComm {
	std::string shmem_name;
	ip::shared_memory_object shmem;
	ip::mapped_region region;
	TuxComm(const std::string & name, bool owner=false): shmem_name(name){
		if (owner) {
		shmem = ip::shared_memory_object(ip::create_only, name.c_str(), ip::read_write);
		shmem.truncate(sizeof(TuxSHMem) + 100);
		} else {
		shmem = ip::shared_memory_object(ip::open_only, name.c_str(), ip::read_write);
		}
		region = ip::mapped_region(shmem, ip::read_write, 0, sizeof(TuxSHMem) + 100);
		}
	~TuxComm() {
		region = ip::mapped_region();
		ip::shared_memory_object::remove(shmem_name.c_str());
	}
	operator TuxSHMem *() {
		return (TuxSHMem*)region.get_address();
	}
	operator const TuxSHMem *() const {
		return (TuxSHMem*)region.get_address();
	}
	const TuxSHMem * get() const {
		return (TuxSHMem*)region.get_address();
	}
	TuxSHMem * get() {
		return (TuxSHMem*)region.get_address();
	}
};
template<bool flip, typename T> void cp(T* out, const T * in, size_t H, size_t W) {
	if (flip) {
		for(size_t i=0; i<H; i++)
			cp<false>(out+i*W, in+(H-i-1)*W, 1, W);
	} else {
		memcpy(out, in, W*H*sizeof(T));
	}
}

#ifdef EXECUTABLE
class TuxSession: public BaseController {
public:
	std::shared_ptr<TuxComm> c;
	int n_sector = 0;
	std::string sector_name;
	TuxSession(std::shared_ptr<TuxComm> c):BaseController(c->get()->acting),c(c) {
	}
	~TuxSession() {
		TuxSHMem * sh = *c;
		sh->quit = true;
		sh->running = false;
	}
	virtual bool should_quit() const override { const TuxSHMem * sh = *c; return sh->quit; }
	virtual bool should_restart() const override { const TuxSHMem * sh = *c; return sh->restart; }
	virtual bool sync() const override { const TuxSHMem * sh = *c; return sh->synchronized; }
	virtual void randomSeed(int * seed) override {
		TuxSHMem * sh = *c;
		if (sh->seed)
			*seed = sh->seed;
		else
			sh->seed = *seed;
	}
	virtual void beginSession() override {
		TuxSHMem * sh = *c;
		sh->step_state = 0;
		sh->running = true;
		sh->restart = false;
		n_sector = 0;
		Sector * sector = GameSession::current()->get_current_sector();
		sector_name = sector->get_name();
	}
	virtual void endSession() override {
		TuxSHMem * sh = *c;
		sh->running = false;
	}
	virtual bool beginFrame(uint64_t frame_id, const Action & action) override  {
		volatile TuxSHMem * sh = *c;
		sh->frame_id = frame_id;
		sh->action = action;
		if (sh->running) {
			// Wait for the step_state == STEP_START (or quit)
			volatile uint8_t & ss = sh->step_state; // Let's be paranoid
			while(ss != TuxSHMem::STEP_START && !sh->quit && !sh->restart);
			if (sh->acting) {
				canAct(true);
				current_action = sh->next_action;
			} else {
				canAct(false);
			}
		}
		return true;
	}
	virtual void fetchFrame(int W, int H, int C, const uint8_t * data) override {
		TuxSHMem * sh = *c;
		sh->imW = W;
		sh->imH = H;
		assert( C == 3 );
		cp<true>(sh->imData, data, H, W*C);
	}
	virtual void fetchLabels(int W, int H, const uint16_t * data) override {
		TuxSHMem * sh = *c;
		sh->lblW = W;
		sh->lblH = H;
		cp<true>(sh->lblData, data, H, W);
	}
	virtual void endFrame() override {
		TuxSHMem * sh = *c;
		
		// Fetch the game state
		Sector * sector = GameSession::current()->get_current_sector();
		Player * player = sector->player;
		if (sector_name != sector->get_name()) {
			sector_name = sector->get_name();
			n_sector += 1;
		}
		
		sh->n_sector = n_sector;
		sh->position = (player->get_bbox().p1.x + player->get_bbox().p2.x)/(2*sector->get_width());
		sh->is_winning = player->is_winning();
		sh->is_invincible = player->is_invincible();
		sh->is_dying = player->is_dying();
		sh->on_ground = player->on_ground();
		sh->bonus = player->get_status()->bonus;
		sh->velocity[0] = player->get_velocity().x;
		sh->velocity[1] = player->get_velocity().y;
		sh->bbox[0] = player->get_bbox().p1.x;
		sh->bbox[1] = player->get_bbox().p1.y;
		sh->bbox[2] = player->get_bbox().p2.x;
		sh->bbox[3] = player->get_bbox().p2.y;
		sh->camera_pos[0] = sector->camera->get_translation().x;
		sh->camera_pos[1] = sector->camera->get_translation().y;
		
		if (sh->lrW && sh->lrH) {
			float px = (sh->bbox[0]+sh->bbox[2])/2 - sh->camera_pos[0], py = sh->bbox[3] - sh->camera_pos[1]; //(sh->bbox[1]+sh->bbox[3])/2
			int x0 = px / SCREEN_WIDTH * sh->lblW - sh->lrW*sh->lrS/2, y0 = py / SCREEN_HEIGHT * sh->lblH - sh->lrS/2 - sh->lrH*sh->lrS/2;
			memset(sh->lrData, 0, sh->lrW*sh->lrH);
			for( int j=0,k=0; j<sh->lrH; j++ )
				for( int i=0; i<sh->lrW; i++, k++ ) {
					uint32_t cnt[NUM_OT] = {0};
					int i0 = x0 + i*sh->lrS, j0 = y0 + j*sh->lrS;
					for(int jj=std::max(j0,0); jj<j0+sh->lrS && jj<sh->lblH; jj++)
						for(int ii=std::max(i0,0); ii<i0+sh->lrS && ii<sh->lblW; ii++)
							cnt[ sh->lblData[jj*sh->lblW+ii]&0xf ]++;
					
					for(int l=0; l<NUM_OT; l++)
						sh->lrCnt[k][l] = 0;//cnt[l];
					
					// Ignore unknown and background
					for(int l=2; l<NUM_OT; l++)
						if (cnt[sh->lrData[k]] < cnt[l] && cnt[l] > sh->lrS * sh->lrS / 4)
							sh->lrData[k] = l;
					sh->lrCnt[k][sh->lrData[k]] = 1;
				}
		}
		
		if (sh->running) {
			sh->step_state = TuxSHMem::STEP_END;
		}
	}
};
int main(int argc, char * argv[]) {
  if (argc < 2) {
    printf("Usage: %s memory_name tux args\n", argv[0]);
    return 1;
  }

  std::shared_ptr<TuxComm> c = std::make_shared<TuxComm>(argv[1]);
  TuxSession controller(c);
  std::vector<const char *> args(1, argv[0]);
  args.insert(args.end(), argv+2, argv+argc);
  Main().run(args.size(), args.data(), &controller);
  
  return 0;
}

#endif

#ifdef PYTHON
std::shared_ptr<TuxComm> start(const std::string & level, int W, int H, bool acting, bool synchronized, bool visible, int lrW=0, int lrH=0, int lrS=0, int seed=0) {
  static int uuid = 1;
  const std::string shmem_name = "pytux" + std::to_string(getpid()) + "_" + std::to_string(uuid++);
	std::shared_ptr<TuxComm> c = std::make_shared<TuxComm>(shmem_name, true);
	{
		TuxSHMem * sh = *c;
		sh->acting = acting;
		sh->synchronized = synchronized;
		sh->lrW = lrW;
		sh->lrH = lrH;
		sh->lrS = lrS;
    sh->seed = seed;
	}
	
	// Fork and start the game
	pid_t pid = fork();
	if (pid != 0) {
		TuxSHMem * sh = *c;
		sh->pid = pid;
		return c;
	}
	
#ifdef __linux__
	// Let the child finish if the parent quits
	prctl(PR_SET_PDEATHSIG, SIGTERM);
	// Just making sure we still have a parent
	if (getppid() == 1)
    	exit(1);
#endif
	
	boost::filesystem::path current_path = boost::filesystem::path(__FILE__).remove_filename();
	execl((current_path/"pytux").c_str(), "pytux", shmem_name.c_str(), level.c_str(), "-g", (std::to_string(W)+"x"+std::to_string(H)).c_str(), "--renderer", visible?"opengl":"egl", NULL);
	((TuxSHMem *)TuxComm(shmem_name))->quit = 1;
	// exec never returns
	_exit(EXIT_FAILURE);
}
static std::unordered_set<pid_t> kill_list;
void killAll() {
	for( pid_t pid: kill_list )
		kill(pid, SIGKILL);
}

class Tux {
protected:
	std::shared_ptr<TuxComm> c;
	Tux(const Tux&) = delete;
	Tux& operator=(const Tux&) = delete;
public:
	Tux(const std::string & level, int W, int H, bool acting=false, bool synchronized=true, bool visible=true, int lrW=0, int lrH=0, int lrS=0, int seed=0) {
		c = start(level, W, H, acting, synchronized, visible, lrW, lrH, lrS, seed);
		TuxSHMem * sh = *c;
		if (sh->pid)
			kill_list.insert(sh->pid);
	}
	~Tux() {
		TuxSHMem * sh = *c;
		quit();
		for(int it=0; it<50 && sh->running; it++) usleep(1000);
		kill(sh->pid, SIGKILL);
	}
	void restart() {
		TuxSHMem * sh = *c;
		sh->restart = true;
	}
	void quit() {
		TuxSHMem * sh = *c;
		sh->quit = true;
	}
	bool waitRunning(){
		TuxSHMem * sh = *c;
		while(!sh->quit && (!sh->running || sh->restart)){ usleep(1000); }

		return sh->running && !sh->restart;
	}
	bool running() {
		TuxSHMem * sh = *c;
		return sh->running && !sh->restart;
	}
	bp::object step(uint32_t action) {
		TuxSHMem * sh = *c;
		if (sh->running) {
			sh->next_action = action;
			// Signal game that it's save to start the next frame
			sh->step_state = TuxSHMem::STEP_START;
			
			// Wait for the frame to finish
			volatile uint8_t & ss = sh->step_state; // Let's be paranoid
			while(ss == TuxSHMem::STEP_START && sh->running && !sh->quit)
				if(PyErr_CheckSignals() == -1) {
					sh->quit = 1;
					bp::throw_error_already_set();
				}
			
			// Fetch the state
			bp::dict state;
			state["n_sector"] = sh->n_sector;
			state["position"] = sh->position;
			state["is_winning"] = sh->is_winning;
			state["is_invincible"] = sh->is_invincible;
			state["is_dying"] = sh->is_dying;
			state["on_ground"] = sh->on_ground;
			state["bonus"] = sh->bonus;
			state["coins"] = sh->coins;
			state["velocity"] = bp::make_tuple(sh->velocity[0], sh->velocity[1]);
			state["bbox"] = bp::make_tuple(sh->bbox[0], sh->bbox[1], sh->bbox[2], sh->bbox[3]);
			
			// Fetch the observation
			bp::dict obs;
			if (sh->imW && sh->imH) {
				npy_intp dims[3] = {sh->imH, sh->imW, 3};
				obs["image"] = boost::python::handle<>(PyArray_SimpleNewFromData(3, dims, NPY_UINT8, sh->imData));
			}
			if (sh->lblW && sh->lblH) {
				npy_intp dims[3] = {sh->lblH, sh->lblW};
				obs["label"] = boost::python::handle<>(PyArray_SimpleNewFromData(2, dims, NPY_UINT16, sh->lblData));
			}
			if (sh->lrW && sh->lrH) {
				npy_intp dims[3] = {sh->lrH, sh->lrW};
				obs["label_lr"] = boost::python::handle<>(PyArray_SimpleNewFromData(2, dims, NPY_UINT8, sh->lrData));
			}
			if (sh->lrW && sh->lrH) {
				npy_intp dims[3] = {sh->lrH, sh->lrW, NUM_OT};
				obs["cnt_lr"] = boost::python::handle<>(PyArray_SimpleNewFromData(3, dims, NPY_UINT16, sh->lrCnt));
			}
			return bp::make_tuple(sh->frame_id, sh->action, state, obs);
		}
		// Set the next action
		return bp::object();
	}
};

BOOST_PYTHON_MODULE(pytux)
{
	using namespace boost::python;
	import_array1();
	atexit(killAll);
	
	class_<Tux, boost::noncopyable>("Tux", init<std::string, int, int, bool, bool, bool, int, int, int, int> ((arg("level"),arg("W"),arg("H"),arg("acting")=false,arg("synchronized")=true,arg("visible")=true,arg("lrW")=0,arg("lrH")=0,arg("lrS")=0,arg("seed")=0)))
	.def("restart", &Tux::restart)
	.def("step", &Tux::step)
	.def("waitRunning", &Tux::waitRunning)
	.def_readonly("running", &Tux::running)
	.def("quit", &Tux::quit);
}
#endif
