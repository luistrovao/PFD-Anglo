import gi

gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gtk, Gdk, Pango

import pandas as pd
import numpy as np
import threading

import copy
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

from filters_module import filtros
from AI_alg_module import algoritmos_AI
from optimization_module import otimizacao


class Manipulador():
    def __init__(self):

        self.estilo()

        self.PFD: Gtk.Stack = Builder.get_object("anglo_w")
        self.Stack: Gtk.Stack = Builder.get_object("stack")
        self.pasta: Gtk.FileChooserDialog = Builder.get_object('local_base')

        self.modelo_armazenamento: Gtk.ListStore = Builder.get_object("liststore1")
        self.lista_entradas: Gtk.ListStore = Builder.get_object("liststore2")
        self.lista_saidas: Gtk.ListStore = Builder.get_object("liststore3")
        self.lista_dt: Gtk.ListStore = Builder.get_object("liststore4")
        self.estima_entrada: Gtk.ListStore = Builder.get_object("liststore5")
        self.estima_saida: Gtk.ListStore = Builder.get_object("liststore6")
        self.define_limites: Gtk.ListStore = Builder.get_object("liststore7")

        self.epocas: Gtk.Entry = Builder.get_object("rna_epochs")

        self.status: Gtk.TextView = Builder.get_object("status")

        self.ap_botao: Gtk.Button = Builder.get_object("filtrar")
        self.av_botao: Gtk.Button = Builder.get_object("avancar")
        self.est_botao: Gtk.Button = Builder.get_object("estimar")
        self.otm_botao: Gtk.Button = Builder.get_object("otimizar")
        self.otm_botao: Gtk.Button = Builder.get_object("otimizar")
        self.tree_botao: Gtk.Button = Builder.get_object("train_tree")
        self.limites_botao: Gtk.Button = Builder.get_object("limites")
        self.voltar1: Gtk.Button = Builder.get_object("voltar1")
        self.avancar: Gtk.Button = Builder.get_object("avanca")
        self.est_botao1: Gtk.Button = Builder.get_object("estimar_dt1")

        self.avancar.set_sensitive(False)

        self.ret_negativo: Gtk.CheckButton = Builder.get_object("ret_negativo")
        self.analise_grafica: Gtk.Button = Builder.get_object("analise_grafica")

        self.BS: Gtk.Entry = Builder.get_object("rna_BS")
        self.layer1: Gtk.Entry = Builder.get_object("rna_1cam")
        self.layer2: Gtk.Entry = Builder.get_object("rna_2cam")
        self.N_cluster: Gtk.Entry = Builder.get_object("n_clusters")
        self.max_depth: Gtk.Entry = Builder.get_object("max_dep")
        self.N_estimator: Gtk.Entry = Builder.get_object("n_estimators")
        self.MW: Gtk.Window = Builder.get_object("main_window")

        ############# VARIÁVEIS DO PFD #######################

        # silos
        self.S1_s1: Gtk.Entry = Builder.get_object("S1_s1")
        self.S1_s2: Gtk.Entry = Builder.get_object("S1_s2")
        self.S1_s3: Gtk.Entry = Builder.get_object("S1_s3")
        self.S2_s1: Gtk.Entry = Builder.get_object("S2_s1")
        self.S2_s2: Gtk.Entry = Builder.get_object("S2_s2")
        self.S2_s3: Gtk.Entry = Builder.get_object("S2_s3")
        self.S3_s1: Gtk.Entry = Builder.get_object("S3_s1")
        self.S3_s2: Gtk.Entry = Builder.get_object("S3_s2")
        self.S3_s3: Gtk.Entry = Builder.get_object("S3_s3")
        self.S4_s1: Gtk.Entry = Builder.get_object("S4_s1")
        self.S4_s2: Gtk.Entry = Builder.get_object("S4_s2")
        self.S5_s1: Gtk.Entry = Builder.get_object("S5_s1")
        self.S5_s2: Gtk.Entry = Builder.get_object("S5_s2")
        self.S5_s3: Gtk.Entry = Builder.get_object("S5_s3")
        self.S6_s1: Gtk.Entry = Builder.get_object("S6_s1")
        self.S6_s2: Gtk.Entry = Builder.get_object("S6_s2")
        self.S6_s3: Gtk.Entry = Builder.get_object("S6_s3")
        self.S7_s1: Gtk.Entry = Builder.get_object("S7_s1")
        self.S7_s2: Gtk.Entry = Builder.get_object("S7_s2")
        self.S7_s3: Gtk.Entry = Builder.get_object("S7_s3")
        self.S8_s1: Gtk.Entry = Builder.get_object("S8_s1")
        self.S8_s2: Gtk.Entry = Builder.get_object("S8_s2")
        self.S9_s1: Gtk.Entry = Builder.get_object("S9_s1")
        self.S9_s2: Gtk.Entry = Builder.get_object("S9_s2")
        self.S9_s3: Gtk.Entry = Builder.get_object("S9_s3")
        self.S10_s1: Gtk.Entry = Builder.get_object("S10_s1")
        self.S10_s2: Gtk.Entry = Builder.get_object("S10_s2")
        self.S11_s1: Gtk.Entry = Builder.get_object("S11_s1")
        self.S11_s2: Gtk.Entry = Builder.get_object("S11_s2")
        self.S11_s3: Gtk.Entry = Builder.get_object("S11_s3")
        self.S12_s1: Gtk.Entry = Builder.get_object("S12_s1")
        self.S12_s2: Gtk.Entry = Builder.get_object("S12_s2")
        self.S12_s3: Gtk.Entry = Builder.get_object("S12_s3")

        # vazamento escória
        self.esc_b1: Gtk.Entry = Builder.get_object("esc_bica1")
        self.esc_b2: Gtk.Entry = Builder.get_object("esc_bica2")
        self.esc_b3: Gtk.Entry = Builder.get_object("esc_bica3")
        self.esc_b4: Gtk.Entry = Builder.get_object("esc_bica4")
        self.esc_b5: Gtk.Entry = Builder.get_object("esc_bica5")
        self.esc_b6: Gtk.Entry = Builder.get_object("esc_bica6")

        # vazamento metal
        self.metal_b3: Gtk.Entry = Builder.get_object("metal_bica3")
        self.metal_b4: Gtk.Entry = Builder.get_object("metal_bica4")

        # temperatura zonas
        self.free_board_T: Gtk.Entry = Builder.get_object("free_board_T")
        self.escoria_T: Gtk.Entry = Builder.get_object("escoria_T")

        # variáveis elétricas
        self.corrente_A: Gtk.Entry = Builder.get_object("correnteA")
        self.potencia_A: Gtk.Entry = Builder.get_object("potenciaA")
        self.corrente_B: Gtk.Entry = Builder.get_object("correnteB")
        self.potencia_B: Gtk.Entry = Builder.get_object("potenciaB")
        self.corrente_C: Gtk.Entry = Builder.get_object("correnteC")
        self.potencia_C: Gtk.Entry = Builder.get_object("potenciaC")
        self.potencia_total: Gtk.Entry = Builder.get_object("Pot_total")

        #################################################################
        self.coluna_LI: Gtk.TreeViewColumn = Builder.get_object("lim_inf")
        self.coluna_LS: Gtk.TreeViewColumn = Builder.get_object("lim_sup")

        self.entradas = []
        self.entradas_val_estima = []
        self.saidas = []
        self.maximas = []
        self.minimas = []
        self.entradas_label = []
        self.saidas_label = []

        self.grafico = Builder.get_object('graf1')
        self.janela_grafico = Builder.get_object('grafico')
        self.canvas = []
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.combo_box: Gtk.ComboBoxText = Builder.get_object("combo_box")
        self.flag_combo = False
        self.N_pontos: Gtk.Entry = Builder.get_object("n_pontos")

        self.Stack.set_visible_child_name("view_inicial")

    def estilo(self):

        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(os.path.join('estilo.css'))
        screen = Gdk.Screen()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen.get_default(),
                                              css_provider,
                                              Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def on_button_login_clicked(self, button):
        email = Builder.get_object("email").get_text()
        senha = Builder.get_object("senha").get_text()
        lembrar = Builder.get_object("lembrar").get_active()
        self.login(email, senha, lembrar)

    def on_main_window_destroy(self, window):
        Gtk.main_quit()

    def mensagem(self, param, param1, param2):
        mensagem: Gtk.MessageDialog = Builder.get_object("mensagem")
        mensagem.props.text = param
        mensagem.props.secondary_text = param1
        mensagem.props.icon_name = param2
        mensagem.show_all()
        mensagem.run()
        mensagem.hide()

    def login(self, email, senha, lembrar):
        if email == 'a' and senha == 'a':
            self.mensagem('Bem vindo', 'Usuario Logado com Sucesso', 'emblem-default')
            self.Stack.set_visible_child_name("view_inicial")
        else:
            self.mensagem('Aviso', 'E-mail ou senha incorretos', 'dialog-error')

    # self.Stack.set_visible_child_name("view_inicial")
    def on_seleciona_base_clicked(self, button):
        self.pasta.show_all()
        response = self.pasta.run()
        if response == Gtk.ResponseType.OK:
            print("File Selected" + self.pasta.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            print('Cancelado')

    def on_button_selecionar_clicked(self, button):
        self.pasta.hide()
        self.Stack.set_visible_child_name('view_inicial')
        self.endereco = Builder.get_object('base_address')
        self.arquivo = self.pasta.get_filename()
        self.endereco.set_text(self.arquivo)

    def on_Input_toggled(self, widget, path):
        # função para marcar/desmarcar checkbox na janela de pré-processamento
        pass
        # self.modelo_armazenamento[path][1] = not self.modelo_armazenamento[path][1]

    def on_Output_toggled(self, widget, path):
        # função para marcar/desmarcar checkbox na janela de pré-processamento
        pass
        # self.modelo_armazenamento[path][2] = not self.modelo_armazenamento[path][2]

    def on_max_dep_changed(self, path):
        self.tree_botao.set_sensitive(True)
        self.lista_dt.clear()

    def on_lim_sup_edited(self, widget, path, text):
        # Alterar os valores dos limites superiores no pré-processamento
        valor = text
        if text.find(',') != -1:
            valor = text.replace(",", ".")
        self.modelo_armazenamento[path][4] = str(float(valor))

    def on_lim_inf_edited(self, widget, path, text):
        # Alterar os valores dos limites inferiores no pré-processamento
        valor = text
        if text.find(',') != -1:
            valor = text.replace(",", ".")
        self.modelo_armazenamento[path][3] = str(float(valor))

    def on_valor_estima_in_edited(self, widget, path, text):
        valor = text
        if text.find(',') != -1:
            valor = text.replace(",", ".")
        self.estima_entrada[path][1] = str(float(valor))

    def on_LS1_edited(self, widget, path, text):
        # Alterar os valores dos limites superiores na otimização
        valor = text
        if text.find(',') != -1:
            valor = text.replace(",", ".")
        self.define_limites[path][2] = str(float(valor))

    def on_LI1_edited(self, widget, path, text):
        # Alterar os valores dos limites inferiores na otimização
        valor = text
        if text.find(',') != -1:
            valor = text.replace(",", ".")
        self.define_limites[path][1] = str(float(valor))

    def on_button_cancelar_clicked(self, button):
        self.pasta.hide()
        self.Stack.set_visible_child_name('view_inicial')

    def on_escolhe_in_out_clicked(self, button):

        self.confirmar.set_label("Base Pré-Selecionada")
        self.confirmar.set_sensitive(True)
        self.avancar.set_sensitive(False)

        self.Stack.set_visible_child_name('view_inicial')

        self.lista_entradas.clear()
        self.lista_saidas.clear()
        self.entradas_label.clear()
        self.saidas_label.clear()
        self.lista_dt.clear()
        self.estima_entrada.clear()
        self.estima_saida.clear()
        self.modelo_armazenamento.clear()

        self.maximas = []
        self.minimas = []

        self.tree_botao.set_sensitive(True)
        self.ap_botao.set_sensitive(True)

    def on_voltar_clicked(self, button):
        self.Stack.set_visible_child_name('view_base')
        self.define_limites.clear()

    def on_voltar_PFD_clicked(self, button):
        self.Stack.set_visible_child_name('view_PFD')
        self.define_limites.clear()

    def on_voltar_train_clicked(self, button):

        self.Stack.set_visible_child_name('view_dt')
        for child in self.grafico:
            child.destroy()

    def on_graf_train_b_clicked(self, button):

        self.Stack.set_visible_child_name("view_graf")

    def on_confirmar_clicked(self, button):

        thread = threading.Thread(target=self.load_base)
        thread.start()

        self.confirmar: Gtk.Button = Builder.get_object("confirmar")
        self.confirmar.set_label("Carregando Base...")
        self.confirmar.set_sensitive(False)
        self.av_botao.set_sensitive(False)
        self.analise_grafica.set_sensitive(False)
        self.limites_botao.set_sensitive(False)

    def load_base(self):

        self.arquivo = os.path.join('Base_anglo.csv')
        self.base = pd.read_csv(self.arquivo, sep=';', engine='python', decimal=",")
        self.base.drop(columns="DateTime", inplace=True)
        aux = self.base.columns.values
        aux = aux.reshape(len(aux), 1)
        i = 0
        aux1 = True
        aux2 = False
        for row in aux:
            if i > 38:
                aux1 = False
                aux2 = True
            self.modelo_armazenamento.append((row[0], aux1, aux2, " ", " "))
            i = i + 1

        self.avancar.set_sensitive(True)

    def on_avanca_clicked(self, button):

        self.Stack.set_visible_child_name("view_variaveis")

    def on_filtrar_clicked(self, button):

        self.entradas.clear()
        self.saidas.clear()

        self.filtro = filtros()

        self.base_c = self.filtro.funcoes["bases_c"](self.base)

        self.base = self.filtro.funcoes['nao_numerico'](self.base)
        self.base_c = self.filtro.funcoes['nao_numerico'](self.base_c)

        if self.ret_negativo.get_active():

            self.base = self.filtro.funcoes['nao-negativo'](self.base)
            self.base_c = self.filtro.funcoes['nao-negativo'](self.base_c)


        base_cp = self.base_c["Potencia_Ativa_Total"]

        self.base_t = self.base.drop(columns=["Potencia_Ativa_Total", "Temp. da Escória"])
        lim_sup, lim_inf = self.filtro.funcoes['quartiles'](self.base_t)

        self.base_esc = self.base.drop(columns=["Temp. Blocos Vazamento Metal ( bica 04)",
                                                "Temp. Blocos Vazamento Metal ( bica 03)",
                                                "Temp. Free Board",
                                                "Temp. Blocos Vazamento Escória ( bica 01)",
                                                "Temp. Blocos Vazamento Escória ( bica 02)",
                                                "Temp. Blocos Vazamento Escória ( bica 03)",
                                                "Temp. Blocos Vazamento Escória ( bica 04)",
                                                "Temp. Blocos Vazamento Escória ( bica 05)",
                                                "Temp. Blocos Vazamento Escória ( bica 06)",
                                                "Potencia_Ativa_Total"], axis=1)

        self.base_aux_esc = self.base_esc.where(self.base_esc["Temp. da Escória"] <= 1700)
        self.base_aux_esc = self.base_aux_esc.where(self.base_aux_esc["Temp. da Escória"] >= 1400)
        self.base_aux_esc.dropna(inplace=True)

        sup, inf = self.filtro.funcoes['quartiles'](self.base_aux_esc["Temp. da Escória"])
        lim_sup = pd.concat([lim_sup, sup], ignore_index=True)
        lim_inf = pd.concat([lim_inf, inf], ignore_index=True)

        sup, inf = self.filtro.funcoes['quartiles'](base_cp)
        lim_sup = pd.concat([lim_sup, sup], ignore_index=True)
        lim_inf = pd.concat([lim_inf, inf], ignore_index=True)

        for row in self.modelo_armazenamento:
            self.entradas.append(row[1])
            self.saidas.append(row[2])

        for i in range(len(self.modelo_armazenamento)):
            if self.entradas[i] == True or self.saidas[i] == True:

                self.modelo_armazenamento[i][4] = str(round(lim_sup[i], 2))

                if lim_inf[i] < 0 and self.ret_negativo.get_active():
                    self.modelo_armazenamento[i][3] = str(0)
                else:
                    self.modelo_armazenamento[i][3] = str(round(lim_inf[i], 2))

        self.ap_botao.set_sensitive(False)
        self.av_botao.set_sensitive(True)

    def on_avancar_clicked(self, button):

        # Preencher estimações com os valores atuais dos inputs

        valores = os.path.join('Val_momento.csv')
        dados = pd.read_csv(valores, sep=';', engine='python', decimal=",")
        dados.drop(columns="DateTime", inplace=True)
        aux = dados.iloc[0, 0:39]

        self.S1_s1.set_text(str(round(aux[0], 1)))
        self.S1_s2.set_text(str(round(aux[1], 1)))
        self.S1_s3.set_text(str(round(aux[2], 1)))
        self.S2_s1.set_text(str(round(aux[3], 1)))
        self.S2_s2.set_text(str(round(aux[4], 1)))
        self.S2_s3.set_text(str(round(aux[5], 1)))
        self.S3_s1.set_text(str(round(aux[6], 1)))
        self.S3_s2.set_text(str(round(aux[7], 1)))
        self.S3_s3.set_text(str(round(aux[8], 1)))
        self.S4_s1.set_text(str(round(aux[9], 1)))
        self.S4_s2.set_text(str(round(aux[10], 1)))
        self.S5_s1.set_text(str(round(aux[11], 1)))
        self.S5_s2.set_text(str(round(aux[12], 1)))
        self.S5_s3.set_text(str(round(aux[13], 1)))
        self.S6_s1.set_text(str(round(aux[14], 1)))
        self.S6_s2.set_text(str(round(aux[15], 1)))
        self.S6_s3.set_text(str(round(aux[16], 1)))
        self.S7_s1.set_text(str(round(aux[17], 1)))
        self.S7_s2.set_text(str(round(aux[18], 1)))
        self.S7_s3.set_text(str(round(aux[19], 1)))
        self.S8_s1.set_text(str(round(aux[20], 1)))
        self.S8_s2.set_text(str(round(aux[21], 1)))
        self.S9_s1.set_text(str(round(aux[22], 1)))
        self.S9_s2.set_text(str(round(aux[23], 1)))
        self.S9_s3.set_text(str(round(aux[24], 1)))
        self.S10_s1.set_text(str(round(aux[25], 1)))
        self.S10_s2.set_text(str(round(aux[26], 1)))
        self.S11_s1.set_text(str(round(aux[27], 1)))
        self.S11_s2.set_text(str(round(aux[28], 1)))
        self.S11_s3.set_text(str(round(aux[29], 1)))
        self.S12_s1.set_text(str(round(aux[30], 1)))
        self.S12_s2.set_text(str(round(aux[31], 1)))
        self.S12_s3.set_text(str(round(aux[32], 1)))
        self.corrente_A.set_text(str(round(aux[33], 1)))
        self.potencia_A.set_text(str(round(aux[34], 1)))
        self.corrente_B.set_text(str(round(aux[35], 1)))
        self.potencia_B.set_text(str(round(aux[36], 1)))
        self.corrente_C.set_text(str(round(aux[37], 1)))
        self.potencia_C.set_text(str(round(aux[38], 1)))

        self.Stack.set_visible_child_name('view_base')

        for i in range(len(self.modelo_armazenamento)):
            if self.entradas[i] == True:
                self.entradas_label.append(self.modelo_armazenamento[i][0])
                self.lista_entradas.append([self.modelo_armazenamento[i][0]])
                self.minimas.append(float(self.modelo_armazenamento[i][3]))
                self.maximas.append(float(self.modelo_armazenamento[i][4]))
                self.estima_entrada.append([self.modelo_armazenamento[i][0], str(round(aux[i], 2))])

            if self.saidas[i] == True:
                self.saidas_label.append(self.modelo_armazenamento[i][0])
                self.lista_saidas.append([self.modelo_armazenamento[i][0]])
                self.minimas.append(float(self.modelo_armazenamento[i][3]))
                self.maximas.append(float(self.modelo_armazenamento[i][4]))
                self.estima_saida.append([self.modelo_armazenamento[i][0], ''])

        ## Trecho para modelo com escolha de entradas/saídas

        self.maximas_c = self.maximas[:39]
        self.maximas_c2 = self.maximas[-1]
        self.maximas_c.append(self.maximas_c2)

        self.minimas_c = self.minimas[:39]
        self.minimas_c2 = self.minimas[-1]
        self.minimas_c.append(self.minimas_c2)

        self.maximas_esc = self.maximas[:39]
        self.maximas_esc2 = self.maximas[-2]
        self.maximas_esc.append(self.maximas_esc2)

        self.minimas_esc = self.minimas[:39]
        self.minimas_esc2 = self.minimas[-2]
        self.minimas_esc.append(self.minimas_esc2)

        self.maximas_t = self.maximas[:48]
        self.minimas_t = self.minimas[:48]

        self.maximas_c = np.asarray(self.maximas_c).reshape(1, len(self.maximas_c))
        self.minimas_c = np.asarray(self.minimas_c).reshape(1, len(self.minimas_c))

        self.maximas_t = np.asarray(self.maximas_t).reshape(1, len(self.maximas_t))
        self.minimas_t = np.asarray(self.minimas_t).reshape(1, len(self.minimas_t))

        self.maximas_esc = np.asarray(self.maximas_esc).reshape(1, len(self.maximas_esc))
        self.minimas_esc = np.asarray(self.minimas_esc).reshape(1, len(self.minimas_esc))

        self.base_aux_t = self.base_t.where(self.base_t <= self.maximas_t)
        self.base_aux_t = self.base_aux_t.where(self.base_aux_t >= self.minimas_t)
        self.base_aux_t.dropna(inplace=True)

        self.base_aux_c = self.base_c.where(self.base_c <= self.maximas_c)
        self.base_aux_c = self.base_aux_c.where(self.base_aux_c >= self.minimas_c)
        self.base_aux_c.dropna(inplace=True)

        self.base_aux_esc = self.base_esc.where(self.base_esc <= self.maximas_esc)
        self.base_aux_esc = self.base_aux_esc.where(self.base_aux_esc >= self.minimas_esc)
        self.base_aux_esc.dropna(inplace=True)

        self.base_aux_cl_t = self.base_aux_t.loc[:, self.entradas_label]
        self.base_aux_cl_c = self.base_aux_c.loc[:, self.entradas_label]
        self.base_aux_cl_esc = self.base_aux_esc.loc[:, self.entradas_label]

        self.n_cl = int(self.N_cluster.get_text())
        [indices_t, self.cluster_dt_t] = self.filtro.funcoes['clusterizacao'](self.n_cl, self.base_aux_cl_t)
        [indices_c, self.cluster_dt_c] = self.filtro.funcoes['clusterizacao'](self.n_cl, self.base_aux_cl_c)
        [indices_esc, self.cluster_dt_esc] = self.filtro.funcoes['clusterizacao'](self.n_cl, self.base_aux_cl_esc)
        ## ALTERAR DEPOIS LINHA ABAIXO

        self.cluster_dt = self.cluster_dt_t

        self.base_aux_t['K classes'] = indices_t
        self.base_aux_c['K classes'] = indices_c
        self.base_aux_esc['K classes'] = indices_esc

    def on_treinar_dt_clicked(self, button):

        self.Stack.set_visible_child_name('view_dt')

        self.combo_box.set_entry_text_column(0)

        if self.flag_combo == False:
            for row in self.saidas_label:
                self.combo_box.append_text(row)
                self.flag_combo = True

    def on_train_tree_clicked(self, button):

        thread = threading.Thread(target=self.treino)
        thread.start()

        self.tree_botao.set_sensitive(False)
        self.voltar1.set_sensitive(False)
        self.textbuffer = self.status.get_buffer()

        self.textbuffer.set_text("TREINAMENTO EM ANDAMENTO...")

    def treino(self):

        self.alg_IA = algoritmos_AI()

        max_dep = int(self.max_depth.get_text())
        n_estim = int(self.N_estimator.get_text())

        self.saidas_label.remove("Potencia_Ativa_Total")
        self.saidas_label.remove("Temp. da Escória")

        [self.dados_dt, self.previsoes_dt, self.resultados_dt, self.modelo_dt] = self.alg_IA.tipos['Dec_Tree'](
            self.base_aux_t, max_dep,
            n_estim, self.n_cl,
            self.entradas_label,
            self.saidas_label)

        [self.dados_dt_esc, self.previsoes_dt_esc, self.resultados_dt_esc, self.modelo_dt_esc] = self.alg_IA.tipos[
            'Dec_Tree'](
            self.base_aux_esc, max_dep,
            n_estim, self.n_cl,
            self.entradas_label,
            ["Temp. da Escória"])

        [self.dados_dt_c, self.previsoes_dt_c, self.resultados_dt_c, self.modelo_dt_c] = self.alg_IA.tipos['Dec_Tree'](
            self.base_aux_c, max_dep,
            n_estim, self.n_cl,
            self.entradas_label,
            ["Potencia_Ativa_Total"])

        self.saidas_label.append("Temp. da Escória")
        self.saidas_label.append("Potencia_Ativa_Total")

        for i in range(len(self.resultados_dt)):
            self.resultados_dt[i] = self.resultados_dt[i] + self.resultados_dt_esc[i] + self.resultados_dt_c[i]
            self.modelo_dt[i] = self.modelo_dt[i] + self.modelo_dt_esc[i] + self.modelo_dt_c[i]
            self.previsoes_dt[i] = self.previsoes_dt[i] + self.previsoes_dt_esc[i] + self.previsoes_dt_c[i]
            self.dados_dt[i] = self.dados_dt[i] + self.dados_dt_esc[i] + self.dados_dt_c[i]

        self.resultados_dt = pd.DataFrame(self.resultados_dt)

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        outdir = './modelos'
        nome_modelo = 'model_DT.pkl'
        nome_previsao = 'previsao_DT.pkl'
        nome_dados = 'dados_DT.pkl'
        nome_resultados = 'resultados_DT.pkl'

        fullname = os.path.join(outdir, nome_modelo)
        fullname_p = os.path.join(outdir, nome_previsao)
        fullname_d = os.path.join(outdir, nome_dados)
        fullname_r = os.path.join(outdir, nome_resultados)

        with open(fullname, 'wb') as file:
            pickle.dump(self.modelo_dt, file)
        with open(fullname_p, 'wb') as file:
            pickle.dump(self.previsoes_dt, file)
        with open(fullname_d, 'wb') as file:
            pickle.dump(self.dados_dt, file)
        with open(fullname_r, 'wb') as file:
            pickle.dump(self.resultados_dt, file)

        self.voltar1.set_sensitive(True)

        self.resultados()

        self.textbuffer.set_text("TREINAMENTO CONCLUÍDO.")

    def on_result_modelo_clicked(self,button):
        self.resultados()

    def resultados(self):

        self.lista_dt.clear()

        outdir = './modelos'
        nome_modelo = 'resultados_DT.pkl'
        fullname = os.path.join(outdir, nome_modelo)

        with open(fullname, 'rb') as file:
            resultados_dt = pickle.load(file)

        i = 0
        while i < len(self.saidas_label):
            self.lista_dt.append((self.saidas_label[i], str(round(np.mean(np.asarray(resultados_dt[:][i])), 2))))
            i += 1

    def on_estimar_clicked(self, button):

        self.Stack.set_visible_child_name("view_PFD")

    def on_mostrar_header_clicked(self, button):
        self.modelo_armazenamento.clear()
        self.modelo_armazenamento.append(('teste'))

    def on_analise_grafica_clicked(self, button):

        for child in self.grafico:
            child.destroy()

        outdir = './modelos'
        nome_modelo = 'previsao_DT.pkl'
        fullname_p = os.path.join(outdir, nome_modelo)
        nome_modelo = 'dados_DT.pkl'
        fullname_d = os.path.join(outdir, nome_modelo)

        with open(fullname_p, 'rb') as file:
            previsoes_dt = pickle.load(file)

        with open(fullname_d, 'rb') as file:
            dados_dt = pickle.load(file)

        self.fig, ax = plt.subplots()
        ax.set_title('Dados Teste vs Dados Previstos')
        ax.set(xlabel='Dados', ylabel=self.combo_box.get_active_text())
        amostras = len(previsoes_dt[0][0])

        if self.lista_saidas[self.combo_box.get_active()][0] == "Potencia_Ativa_Total":
            Y1 = dados_dt[0][5].loc[:, self.lista_saidas[self.combo_box.get_active()][0]]
        elif self.lista_saidas[self.combo_box.get_active()][0] == "Temp. da Escória":
            Y1 = dados_dt[0][3].loc[:, self.lista_saidas[self.combo_box.get_active()][0]]
        else:
            Y1 = dados_dt[0][1].loc[:, self.lista_saidas[self.combo_box.get_active()][0]]

        n_pts = int(self.N_pontos.get_text())

        if len(Y1) < n_pts:
            X = range(len(Y1))
        else:
            X = range(n_pts)

        Y2 = previsoes_dt[0][self.combo_box.get_active()]
        Z1 = Y1[0:n_pts]
        Z2 = Y2[0:n_pts]
        ax.plot(X, Z1, "ro-", label="Dados Reais", linewidth=1)
        ax.plot(X, Z2, "bo--", label="Dados do Modelo", linewidth=1)
        ax.grid(True)
        ax.legend()

        if self.canvas:
            self.canvas.flush_events()
            self.canvas = FigureCanvas(self.fig)
            self.canvas.draw()
            self.canvas.show()
            self.grafico.add(self.canvas)
        else:
            self.canvas = FigureCanvas(self.fig)
            self.canvas.show()
            self.grafico.add(self.canvas)

    def on_combo_box_changed(self, combo):
        self.texto_grafico = combo.get_active_text()
        self.analise_grafica.set_sensitive(True)

    def on_estimar_dt_clicked(self, button):

        thread = threading.Thread(target=self.estima_paralelo)
        thread.start()

        self.metal_b3.set_text("")
        self.metal_b4.set_text("")
        self.free_board_T.set_text("")
        self.esc_b1.set_text("")
        self.esc_b2.set_text("")
        self.esc_b3.set_text("")
        self.esc_b4.set_text("")
        self.esc_b5.set_text("")
        self.esc_b6.set_text("")
        self.escoria_T.set_text("")
        self.potencia_total.set_text("")

    def estima_paralelo(self):

        outdir = './modelos'
        nome_modelo = 'model_DT.pkl'
        fullname = os.path.join(outdir, nome_modelo)

        with open(fullname, 'rb') as file:
            self.modelo_dt = pickle.load(file)

        self.entradas_val_estima.clear()

        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S1_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S1_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S1_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S2_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S2_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S2_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S3_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S3_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S3_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S4_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S4_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S5_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S5_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S5_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S6_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S6_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S6_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S7_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S7_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S7_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S8_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S8_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S9_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S9_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S9_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S10_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S10_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S11_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S11_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S11_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S12_s1.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S12_s2.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.S12_s3.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.corrente_A.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.potencia_A.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.corrente_B.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.potencia_B.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.corrente_C.get_text())))
        self.entradas_val_estima.append(float(self.filtro.funcoes['retira_virgula'](self.potencia_C.get_text())))

        a = pd.DataFrame(self.entradas_val_estima).T

        cl_n = self.cluster_dt.predict(a)

        for i in range(len(self.estima_saida)):
            x = self.modelo_dt[int(cl_n)][i]
            self.estima_saida[i][1] = str(round(float(x.predict(a)), 1))

        self.metal_b3.set_text(self.estima_saida[0][1])
        self.metal_b4.set_text(self.estima_saida[1][1])
        self.free_board_T.set_text(self.estima_saida[2][1])
        self.esc_b1.set_text(self.estima_saida[3][1])
        self.esc_b2.set_text(self.estima_saida[4][1])
        self.esc_b3.set_text(self.estima_saida[5][1])
        self.esc_b4.set_text(self.estima_saida[6][1])
        self.esc_b5.set_text(self.estima_saida[7][1])
        self.esc_b6.set_text(self.estima_saida[8][1])
        self.escoria_T.set_text(self.estima_saida[9][1])
        self.potencia_total.set_text(self.estima_saida[10][1])

    def on_otimizar_clicked(self, button):

        inp_manipuladas = ["Corrente Transformador A ", "Corrente Transformador B ", "Corrente Transformador C ",
                           "Potência Transformador A", "Potência Transformador B", "Potência Transformador C"]

        for i in range(len(inp_manipuladas)):
            self.define_limites.append([inp_manipuladas[i], str(16.0), str(20.0)])

        self.Stack.set_visible_child_name("view_limites")

    def on_limites_clicked(self, button):
        lb = []
        ub = []

        for i in range(len(self.define_limites)):
            lb.append(self.define_limites[i][1])
            ub.append(self.define_limites[i][2])

        inputs = self.base_t
        # matriz horário x0 = [25,25,25,20,20,20]
        otm = otimizacao(inputs, self.cluster_dt, self.modelo_dt, x0, lb, ub)

        x = otm.methods['minimizar']


Builder = Gtk.Builder()
Builder.add_from_file("user_interface.glade")
Builder.connect_signals(Manipulador())
Window: Gtk.Window = Builder.get_object("main_window")
Window.show_all()
Gtk.main()
