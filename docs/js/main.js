"use strict";

/*

View constructor

*/
var View = function(name) {
  this.enabled = false;
  this.name = name;
  this.node = document.querySelector(".view-" + name);
  this.enableButton = document.querySelector(".enable-" + name);
  this.disableButtons = document.querySelectorAll(".enable-view:not(.enable-" + name + ")");
  
  this.enable = function() {
    this.node.classList.remove("is-hidden");
    this.enableButton.classList.add("is-active");
    this.enabled = true;
    if (this.onenable)
      this.onenable();
    return this;
  };

  this.disable = function() {
    this.node.classList.add("is-hidden");
    this.enableButton.classList.remove("is-active");
    this.enabled = false;
    if (this.ondisable)
      this.ondisable();
    return this;
  }

  this.enableButton.addEventListener("click", this.enable.bind(this));
  for (let button of this.disableButtons) {
    button.addEventListener("click", this.disable.bind(this));
  }
};

/*

Task constructor

*/
var Task = function() {
  
};

/*

Learner constructor

*/
var Learner = function() {
  this.type = null;
  this.nodeType = document.createElement("select");

  this.activation = null;
  this.nodeActivation = document.createElement("select");

  this.hidden = null;
  this.nodesHidden = [document.createElement("input")];
  
  this.rounds = 0;
  this.nodeRounds = document.createElement("input");
};

/*

Visualization constructor

*/

const singleTemplate = `
      <div class="columns">
        <div class="column is-narrow">
          <h1 class="title">Visualization {index}</h1>
          <h1 id="title{index}" class="shiny-text-output">Emptiness :(</h1>
          
          <aside class="menu">
            <p class="menu-label">Data</p>
            <form>
              <div class="field">
                <label for="datasetId{index}" class="label">Current dataset</label>
                <p class="control">
                  <output id="datasetId{index}" class="shiny-text-output"></output>
                </p>
              </div>
              
              <div class="field">
                <label for="taskData{index}" class="button is-primary">Upload new dataset</label>
                <p class="control">
                  <input id="taskData{index}" name="taskData{index}" class="input is-hidden" type="file" />
                </p>
              </div>
              
              <div class="field">
                <label for="taskCl{index}" class="label">Class attribute</label>
                <p class="control">
                  <div class="select">
                    <select id="taskCl{index}" name="taskCl{index}"></select>
                  </div>
                </p>
              </div>
            </form>

            <hr>
            <p class="menu-label">Learner</p>
            <form>
              <div class="field">
                <label for="learnerCl{index}" class="label">Type</label>
                <p class="control">
                  <div class="select">
                    <select id="learnerCl{index}" name="learnerCl{index}"></select>
                  </div>
                </p>
              </div>
              <div class="field">
                <span class="label">Network layer sizes</span>
                <p class="control">
                  <output id="learnerFirst{index}" class="shiny-text-output"></output>
                  <input name="middleLayer{index}" type="number" class="input" min=1 value=2>
                  <output id="learnerLast{index}" class="shiny-text-output"></output>
                </p>
              </div>
              <div class="field">
                <span class="label">Activation type</span>
                <p class="control">
                  <div class="select">
                    <select id="learnerAct{index}" name="learnerAct{index}"></select>
                  </div>
                </p>
              </div>
            </form>

            <hr>
            <p class="menu-label">Training</p>
            <form>
              <div class="field">
                <span class="label">Optimizer</span>
                <p class="control">
                  <div class="select">
                    <select id="learnerOpt{index}" name="learnerOpt{index}"></select>
                  </div>
                </p>
              </div>
              <div class="field">
                <span class="label">Number of rounds (epochs)</span>
                <p class="control">
                  <input type="number" class="input" min=1 value=10 id="learnerRounds{index}" name="learnerRounds{index}">
                </p>
              </div>
              <div class="field">
                <span class="label">Momentum</span>
                <p class="control">
                  <input type="text" class="input" id="learnerMomentum{index}" name="learnerMomentum{index}">
                </p>
              </div>
              <div class="field">
                <span class="label">Learning rate</span>
                <p class="control">
                  <input type="text" class="input" id="learnerRate{index}" name="learnerRate{index}">
                </p>
              </div>
            </form>

            <hr>
            <p class="menu-label">Visualization</p>
            <form>
            </form>
          </aside>
        </div>

        <div class="column">
          <figure class="highlight">
            {plotOutput}
          </figure>
          <pre id="console{index}" class="shiny-text-output"></pre>
        </div>
      </div>
`;

const gridTemplate = `
          <div class="card">
            <div class="card-image">
              <figure class="image is-16by9">
                {plotOutput}
              </figure>
            </div>
            <div class="card-content">
              <div class="media">
                <div class="media-content">
                  <p class="title is-4">{title}</p>
                  <p class="subtitle is-6">{learnerType}</p>
                </div>
              </div>

              <div class="content">
                Parámetros blablabla....
              </div>
            </div>
          </div>
`;

const listTemplate = `
      <section class="section">
        <div class="container">
          <div class="hero">
            <h1 class="title">{title}</h1>
            <h2 class="subtitle">{learnerType}</h2>
          </div>
          <p>Parámetros...</p>
          <figure class="image is-16by9">
            {plotOutput}
          </figure>
        </div>
      </section>
`;

var Visualization = function(index, task, learner) {

  var templatify = function(template, params) {
    for (var p in params) {
      var val = params[p];
      var rgx = new RegExp("\\{" + p + "\\}", "g");
      console.log(rgx, val);
      template = template.replace(rgx, val);
    }

    return template;
  }
  
  this.task = task;
  this.learner = learner;
  this.selected = false;

  this.singleNode = document.createElement("div");
  this.gridNode = document.createElement("div");
  this.listNode = document.createElement("div");

  this.container = function(node) {
    var c = document.createElement("div");
    c.classList.add("template-container");
    c.setAttribute("data-id", node.id);
    return c;
  };
  
  this.plotNode = document.createElement("div");
  this.plotNode.id = "bigPlot" + (index + 1);
  this.plotNode.classList.add("plotly");
  this.plotNode.classList.add("html-widget");
  this.plotNode.classList.add("html-widget-output");
  
  this.populate = function() {
    var params = {
      "index": index + 1,
      "plotOutput": this.container(this.plotNode).outerHTML
    };
    
    this.singleNode.innerHTML = templatify(singleTemplate, params);
    this.gridNode.innerHTML = templatify(gridTemplate, params);
    this.listNode.innerHTML = templatify(listTemplate, params);
  };

  this.populate();

  //this.plotNode = this.singleNode.querySelector("#bigPlot" + index);

  this.single = function() {
    // error: this.plotNode es null aquí :(
    this.singleNode.querySelector("[data-id=\"" + this.plotNode.id + "\"]").appendChild(this.plotNode);
    return this.singleNode;
  }
  this.grid = function() {
    this.gridNode.querySelector("[data-id=\"" + this.plotNode.id + "\"]").appendChild(this.plotNode);
    return this.gridNode;
  }
  this.list = function() {
    this.listNode.querySelector("[data-id=\"" + this.plotNode.id + "\"]").appendChild(this.plotNode);
    return this.listNode;
  }
};

/*

Visualization management

*/

var _VisHandler = function() {
  var views = {
    "start": new View("start"),
    "one": new View("one"),
    "grid": new View("grid"),
    "list": new View("list")
  };
  views["start"].enable();
  views["one"].disable();
  views["grid"].disable();
  views["list"].disable();
  
  var visualizations = [];
  var tabs = [];
  var tabsNode = document.querySelector(".numbered-tabs");
  var plusNode = tabsNode.querySelector(".new-visualization");

  // Manage changing elements from view to view
  // Elements need to be transported from one view to another, especially
  // plot elements, so we do this instead of copying them
  for (var name in views) {
    var view = views[name];
    if (name != "start") {
      view.onenable = function() {
        var column = false;
        for (var vis of visualizations) {
          if (this.name == "one") {
            if (vis.selected) {
              if (this.node.lastChild) {
                this.node.removeChild(this.node.lastChild);
              }
              
              this.node.appendChild(vis.single());
            }
          } else if (this.name == "grid") {
            this.node.querySelector(".grid-column-" + Number(column)).appendChild(vis.grid());
            column = !column;
          } else {
            this.node.appendChild(vis.list());
          }
        }
      };
    }
  }

  views["one"].ondisable = function() {
    try {
      tabsNode.querySelector(".is-active").classList.remove("is-active");
    } catch (ex) {}
  };
  
  var select = function(index) {
    if (index > -1 && index < visualizations.length) {
      Shiny.unbindAll();

      for (var name in views) {
        views[name].disable();
      }
      views["one"].enable();
      
      if (views["one"].node.hasChildNodes())
        views["one"].node.removeChild(views["one"].node.lastChild);

      views["one"].node.appendChild(visualizations[index].single());
      
      var prev = tabsNode.querySelector(".is-active");
      if (prev)
        prev.classList.remove("is-active");
      tabs[index].classList.add("is-active");

      visualizations[index].selected = true;
      
      Shiny.onInputChange("currentVis", index + 1);
      Shiny.bindAll();
    }
  };
  
  var newTab = function(index) {
    // <a class="nav-item is-tab is-active">
    //   {index}
    // </a>
    var node = document.createElement("a");
    node.classList.add("nav-item");
    node.classList.add("is-tab");
    node.textContent = index + 1;
    node.addEventListener("click", (function() { select(index); }));
    return node;
  };

  var push = function(vis) {
    visualizations.push(vis);
    var tab = newTab(tabs.length);
    tabs.push(tab);
    tabsNode.insertBefore(tab, plusNode);
  };

  return {
    add: function() {
      Shiny.unbindAll();
      push(new Visualization(visualizations.length, new Task(), new Learner()));
      Shiny.onInputChange("visCount", visualizations.length);
      Shiny.bindAll();
      
      select(visualizations.length - 1);
    }
  };
};


window.addEventListener("load", function() {
  var VisHandler = _VisHandler();
  
  var plusNodes = document.querySelectorAll(".new-visualization");
  plusNodes.forEach(function(p) {
      p.addEventListener("click", function() {
        VisHandler.add();
      });
  });

  Shiny.addCustomMessageHandler("bigPlot", function(plot) {
    console.log(plot);
    document.body.appendChild(plot);
  });

  
});

